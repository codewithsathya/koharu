//! Headless `translate` subcommand.
//!
//! Resolves one input (folder of images / existing project id / `.khr`
//! archive) into a project, loads a local translator LLM, then runs the full
//! pipeline **one page at a time** so each rendered page can be written to the
//! output folder the moment it finishes — no GUI, no HTTP server.

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use camino::{Utf8Path, Utf8PathBuf};
use koharu_app::pipeline::{
    self, PipelineRunOptions, PipelineSpec, ProgressSink, ProgressTick, Scope, WarningSink,
    WarningTick,
};
use koharu_app::{App, AppConfig, ProjectSession, archive, config as app_config, import, projects};
use koharu_core::{ImageRole, LlmStateStatus, NodeKind, PageId};
use koharu_llm::ModelId;
use koharu_runtime::{ComputePolicy, RuntimeHttpConfig, RuntimeManager};
use tokio::sync::broadcast;

use crate::cli::TranslateArgs;

pub(crate) async fn run(args: &TranslateArgs, cpu: bool) -> Result<()> {
    let config: AppConfig = app_config::load()?;
    let runtime = Arc::new(build_runtime(&config, cpu)?);
    runtime.prepare().await.context("prepare runtime")?;
    let app = Arc::new(App::new(
        config.clone(),
        runtime,
        cpu,
        crate::version::current(),
    )?);
    koharu_llm::suppress_native_logs();

    // 1. Resolve the input into an open session.
    let session = open_input(&app, &config, args).await?;
    let total = session.scene.read().pages.len();
    if total == 0 {
        bail!("project has no pages to translate");
    }
    println!(
        "translating {total} page(s) with {} → {}",
        args.model, args.target_lang
    );

    // 2. Load the translator model and wait until it is ready.
    load_model(&app, &args.model).await?;

    // 3. Run every configured stage; build_order resolves execution order.
    let steps = pipeline_steps(&config, args.no_font_match);

    // 4. Prepare the optional output directory.
    let out_dir = args.output.as_deref().map(Utf8PathBuf::from);
    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("create output dir {dir}"))?;
    }

    // 5. Translate page-by-page so output streams as each page completes.
    let page_ids: Vec<PageId> = session.scene.read().pages.keys().copied().collect();
    let cancel = Arc::new(AtomicBool::new(false));
    let mut failures = 0usize;
    let mut skipped = 0usize;
    let run_started = Instant::now();
    for (i, pid) in page_ids.iter().enumerate() {
        let label = page_label(&session, *pid, i);

        // Resume: a page already rendered (in the persisted scene) or already
        // written to the output dir needs no recompute. Re-export it if the
        // output file is missing, then move on.
        if args.resume && page_already_done(&session, *pid, out_dir.as_ref(), &label) {
            let written = write_output(&session, *pid, out_dir.as_ref(), &label)?;
            println!(
                "\n[{:>2}/{}] {label}  ✓ already done{written}",
                i + 1,
                total
            );
            skipped += 1;
            continue;
        }

        println!("\n[{:>2}/{}] {label}", i + 1, total);

        let spec = PipelineSpec {
            scope: Scope::Pages(vec![*pid]),
            steps: steps.clone(),
            options: PipelineRunOptions {
                target_language: Some(args.target_lang.clone()),
                ..Default::default()
            },
        };
        let page_started = Instant::now();
        let outcome = pipeline::run(
            session.clone(),
            app.registry.clone(),
            app.runtime.clone(),
            cpu,
            app.llm.clone(),
            app.renderer.clone(),
            spec,
            cancel.clone(),
            Some(stage_timing_sink()),
            Some(warning_sink()),
        )
        .await?;
        failures += outcome.warning_count;

        // Footer: page wall-clock + how many text blocks were translated, and the
        // written filename when an output dir was given.
        let secs = page_started.elapsed().as_secs_f64();
        let blocks = count_text_blocks(&session, *pid);
        let written = write_output(&session, *pid, out_dir.as_ref(), &label)?;
        println!("        ── {secs:.1}s · {blocks} blocks{written}");
    }

    // 6. Flush + release the project (also persists the translated scene).
    app.close_project().await?;

    let elapsed = run_started.elapsed().as_secs_f64();
    let translated = total - skipped;
    let avg = if translated > 0 {
        elapsed / translated as f64
    } else {
        0.0
    };
    let mut extra = String::new();
    if skipped > 0 {
        extra.push_str(&format!(" · {skipped} skipped"));
    }
    if failures > 0 {
        extra.push_str(&format!(" · {failures} step failure(s)"));
    }
    println!(
        "\ndone · {total} page(s) · {} · avg {avg:.1}s/page{extra}",
        fmt_hms(elapsed)
    );
    Ok(())
}

fn build_runtime(config: &AppConfig, cpu: bool) -> Result<RuntimeManager> {
    let http = RuntimeHttpConfig {
        connect_timeout_secs: config.http.connect_timeout.max(1),
        read_timeout_secs: config.http.read_timeout.max(1),
        max_retries: config.http.max_retries,
    };
    let compute = if cpu {
        ComputePolicy::CpuOnly
    } else {
        ComputePolicy::PreferGpu
    };
    RuntimeManager::new_with_http(config.data.path.as_std_path(), compute, http)
}

/// Open the session described by the (clap-validated, mutually exclusive) input
/// args. Mirrors the `projects` rpc routes' create/open/import sequences.
async fn open_input(
    app: &App,
    config: &AppConfig,
    args: &TranslateArgs,
) -> Result<Arc<ProjectSession>> {
    if let Some(folder) = &args.folder {
        let folder = Utf8PathBuf::from(folder);
        if !folder.is_dir() {
            bail!("not a folder: {folder}");
        }
        let images = collect_images(&folder)?;
        if images.is_empty() {
            bail!("no images found in {folder}");
        }
        let name = args
            .name
            .clone()
            .or_else(|| folder.file_name().map(str::to_string))
            .unwrap_or_else(|| "Untitled".to_string());

        if args.resume {
            // Reuse a stable project keyed by name; import only images not yet
            // present so an interrupted run picks up where it left off.
            let session = open_or_create(app, config, &name, &name).await?;
            let have: HashSet<String> = session
                .scene
                .read()
                .pages
                .values()
                .map(|p| p.name.clone())
                .collect();
            let fresh: Vec<Utf8PathBuf> = images
                .into_iter()
                .filter(|p| !have.contains(p.file_name().unwrap_or_default()))
                .collect();
            if have.is_empty() {
                let n = import::import_image_files(&session, &fresh)?.len();
                println!("imported {n} image(s)");
            } else {
                let n = import::import_image_files(&session, &fresh)?.len();
                println!(
                    "resuming '{name}': {} existing, {n} new page(s)",
                    have.len()
                );
            }
            return Ok(session);
        }

        // allocate_named creates the dir; ProjectSession::create needs it absent.
        let dir = projects::allocate_named(config, &name)?;
        std::fs::remove_dir_all(dir.as_std_path()).ok();
        let session = app.open_project(dir, Some(name)).await?;
        let n = import::import_image_files(&session, &images)?.len();
        println!("imported {n} image(s)");
        Ok(session)
    } else if let Some(id) = &args.project_id {
        // Existing project: scene (and its rendered pages) already persist, so
        // resume here is just the per-page skip in the run loop.
        let path = projects::project_path(config, id)?;
        if !path.exists() {
            bail!("project not found: {id}");
        }
        app.open_project(path, None).await
    } else if let Some(khr) = &args.khr {
        let khr = Utf8PathBuf::from(khr);
        if !khr.is_file() {
            bail!("khr file not found: {khr}");
        }
        if args.resume {
            // Stable project keyed by the archive stem; import the archive only
            // the first time, then reuse the translated project on later runs.
            let stem = khr.file_stem().unwrap_or("imported");
            let dir = projects::project_path(config, stem)?;
            if dir.join("project.toml").exists() {
                println!("resuming imported project '{stem}'");
                return app.open_project(dir, None).await;
            }
            archive::import_khr(&khr, &dir)?;
            return app.open_project(dir, None).await;
        }
        let dir = projects::allocate_imported(config, khr.file_stem())?;
        std::fs::remove_dir_all(dir.as_std_path()).ok();
        archive::import_khr(&khr, &dir)?;
        app.open_project(dir, None).await
    } else {
        // Unreachable: clap's ArgGroup guarantees exactly one input.
        bail!("no input given");
    }
}

/// Open a stable project by `id_hint`, creating it (named `name`) on first use.
/// A `project.toml` marks an already-initialised project.
async fn open_or_create(
    app: &App,
    config: &AppConfig,
    id_hint: &str,
    name: &str,
) -> Result<Arc<ProjectSession>> {
    let dir = projects::project_path(config, id_hint)?;
    if dir.join("project.toml").exists() {
        app.open_project(dir, None).await
    } else {
        app.open_project(dir, Some(name.to_string())).await
    }
}

fn collect_images(folder: &Utf8Path) -> Result<Vec<Utf8PathBuf>> {
    let mut out = Vec::new();
    for entry in
        std::fs::read_dir(folder.as_std_path()).with_context(|| format!("read dir {folder}"))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        if let Ok(p) = Utf8PathBuf::from_path_buf(entry.path())
            && import::is_image_path(&p)
        {
            out.push(p);
        }
    }
    Ok(out)
}

/// Load a local translator model by id and block until it is ready (the load is
/// fire-and-forget, so we watch the state broadcast).
// ponytail: no download-progress bar; the model downloads silently on first
// use. Add a runtime.subscribe_downloads() printer if users want a progress UI.
async fn load_model(app: &App, model: &str) -> Result<()> {
    let id: ModelId = model.parse().map_err(|_| {
        anyhow::anyhow!(
            "unknown local translator model '{model}'. Use a local model id from the README \
             'Translation models' section (e.g. 'lfm2.5-1.2b-instruct', 'qwen3.5-2b', \
             'sakura-galtransl-7b-v3.7')."
        )
    })?;
    print!("loading model '{model}' (downloads on first use)… ");
    use std::io::Write;
    std::io::stdout().flush().ok();
    let started = Instant::now();

    let mut rx = app.llm.subscribe();
    app.llm.load_local(id).await;
    loop {
        let snap = app.llm.snapshot().await;
        match snap.status {
            LlmStateStatus::Ready => {
                println!("ready ({:.1}s)", started.elapsed().as_secs_f64());
                return Ok(());
            }
            LlmStateStatus::Failed => {
                bail!(
                    "model '{model}' failed to load: {}",
                    snap.error.unwrap_or_default()
                );
            }
            _ => {}
        }
        match rx.recv().await {
            Ok(_) => {} // state changed — re-check the snapshot
            Err(broadcast::error::RecvError::Lagged(_)) => {}
            Err(broadcast::error::RecvError::Closed) => {
                if app.llm.ready().await {
                    return Ok(());
                }
                bail!("llm state channel closed before model '{model}' became ready");
            }
        }
    }
}

/// The configured engine for every pipeline stage. `build_order` topologically
/// sorts them by artifact dependencies, so input order is irrelevant.
/// `no_font_match` drops the font-detection stage (renderer falls back to
/// default styling).
fn pipeline_steps(config: &AppConfig, no_font_match: bool) -> Vec<String> {
    let p = &config.pipeline;
    let mut steps = vec![
        p.detector.clone(),
        p.segmenter.clone(),
        p.bubble_segmenter.clone(),
        p.ocr.clone(),
        p.translator.clone(),
        p.inpainter.clone(),
        p.renderer.clone(),
    ];
    if !no_font_match {
        steps.push(p.font_detector.clone());
    }
    steps
}

fn page_label(session: &ProjectSession, pid: PageId, index: usize) -> String {
    session
        .scene
        .read()
        .pages
        .get(&pid)
        .map(|p| p.name.clone())
        .unwrap_or_else(|| format!("page-{:03}", index + 1))
}

fn output_name(label: &str) -> String {
    let stem = Utf8Path::new(label).file_stem().unwrap_or(label);
    format!("{stem}.png")
}

/// Write a page's rendered PNG into `out_dir` (if given) and return a footer
/// suffix (` → name`). When the page has no rendered node we keep any file a
/// prior run already produced rather than clobbering it.
fn write_output(
    session: &Arc<ProjectSession>,
    pid: PageId,
    out_dir: Option<&Utf8PathBuf>,
    label: &str,
) -> Result<String> {
    let Some(dir) = out_dir else {
        return Ok(String::new());
    };
    let name = output_name(label);
    match koharu_rpc::psd_export::png_bytes_for_page(session, pid, ImageRole::Rendered)? {
        Some(png) => {
            std::fs::write(dir.join(&name).as_std_path(), png)
                .with_context(|| format!("write {dir}/{name}"))?;
            Ok(format!(" → {name}"))
        }
        None if dir.join(&name).exists() => Ok(format!(" → {name} (kept)")),
        None => Ok(" → (no rendered output)".to_string()),
    }
}

/// True when a page needs no recompute: it already has a rendered image node
/// (persisted in the scene) or its output PNG is already on disk.
fn page_already_done(
    session: &ProjectSession,
    pid: PageId,
    out_dir: Option<&Utf8PathBuf>,
    label: &str,
) -> bool {
    if has_rendered_node(session, pid) {
        return true;
    }
    out_dir.is_some_and(|dir| dir.join(output_name(label)).as_std_path().exists())
}

fn has_rendered_node(session: &ProjectSession, pid: PageId) -> bool {
    session.scene.read().pages.get(&pid).is_some_and(|p| {
        p.nodes
            .values()
            .any(|n| matches!(&n.kind, NodeKind::Image(d) if d.role == ImageRole::Rendered))
    })
}

/// Per-page progress sink that prints each stage with its duration. A tick fires
/// just before each step runs (and a final step-less tick at 100%), so the gap
/// between consecutive ticks is the previous stage's wall-clock time. Fresh
/// state per call → durations reset per page.
fn stage_timing_sink() -> ProgressSink {
    let prev: Arc<Mutex<Option<(String, Instant)>>> = Arc::new(Mutex::new(None));
    Arc::new(move |tick: ProgressTick| {
        let mut guard = prev.lock().unwrap();
        // Close the stage that just finished.
        if let Some((id, started)) = guard.take() {
            println!(
                "        {:<14}{:>6.1}s",
                stage_name(&id),
                started.elapsed().as_secs_f64()
            );
        }
        // Open the next stage (the final tick has an empty step_id).
        if !tick.step_id.is_empty() {
            *guard = Some((tick.step_id.clone(), Instant::now()));
        }
    })
}

fn warning_sink() -> WarningSink {
    Arc::new(|tick: WarningTick| {
        println!(
            "        ⚠ {} failed: {}",
            stage_name(&tick.step_id),
            tick.message
        );
    })
}

/// Friendly label for an engine id. Falls back to the raw id for unknown
/// engines (e.g. a non-default model swapped in via config).
fn stage_name(id: &str) -> &str {
    match id {
        "speech-bubble-segmentation" => "bubble-detect",
        "pp-doclayout-v3" => "layout",
        "comic-text-detector-seg" => "text-detect",
        "yuzumarker-font-detection" => "font-detect",
        "lama-manga" => "inpaint",
        "koharu-renderer" => "render",
        "llm" => "translate",
        other if other.contains("ocr") => "ocr",
        other => other,
    }
}

/// Number of translated text blocks on a page (Text nodes).
fn count_text_blocks(session: &ProjectSession, pid: PageId) -> usize {
    session
        .scene
        .read()
        .pages
        .get(&pid)
        .map(|p| {
            p.nodes
                .values()
                .filter(|n| matches!(n.kind, NodeKind::Text(_)))
                .count()
        })
        .unwrap_or(0)
}

/// Whole-run elapsed as `9m56s` / `42s`.
fn fmt_hms(secs: f64) -> String {
    let s = secs.round() as u64;
    let (m, s) = (s / 60, s % 60);
    if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}
