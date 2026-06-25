use clap::{ArgGroup, Args, Parser, Subcommand};

#[derive(Parser)]
#[command(version = crate::version::APP_VERSION, about)]
pub(crate) struct Cli {
    #[arg(short, long, help = "Download dynamic libraries and exit")]
    pub(crate) download: bool,
    #[arg(long, help = "Force CPU even if GPU is available")]
    pub(crate) cpu: bool,
    #[arg(short, long, value_name = "PORT", help = "Bind to a specific port")]
    pub(crate) port: Option<u16>,
    #[arg(
        long,
        help = "Bind the HTTP service to a specific host instead of 127.0.0.1"
    )]
    pub(crate) host: Option<String>,
    #[arg(long, help = "Run without GUI")]
    pub(crate) headless: bool,
    #[arg(long, help = "Enable debug console output")]
    pub(crate) debug: bool,
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Translate manga pages end-to-end without a GUI.
    Translate(TranslateArgs),
}

/// Exactly one input source is required: `--folder`, `--project-id`, or `--khr`.
#[derive(Args)]
#[command(group(
    ArgGroup::new("input").required(true).args(["folder", "project_id", "khr"])
))]
pub(crate) struct TranslateArgs {
    /// Folder of images: create a new project and translate it.
    #[arg(long, value_name = "DIR")]
    pub(crate) folder: Option<String>,
    /// Existing koharu project id to open and translate.
    #[arg(long, value_name = "ID")]
    pub(crate) project_id: Option<String>,
    /// Path to a `.khr` archive to import and translate.
    #[arg(long, value_name = "FILE")]
    pub(crate) khr: Option<String>,
    /// Translator LLM model id (local llama.cpp model, e.g. "qwen3.5-2b").
    #[arg(long, value_name = "MODEL", default_value = "vntl-llama3-8b-v2")]
    pub(crate) model: String,
    /// Target language for translation.
    #[arg(long, value_name = "LANG", default_value = "english")]
    pub(crate) target_lang: String,
    /// Directory to write each translated page (PNG) into as it completes.
    #[arg(long, value_name = "DIR")]
    pub(crate) output: Option<String>,
    /// Name for the new project (with `--folder`; defaults to the folder name).
    #[arg(long, value_name = "NAME")]
    pub(crate) name: Option<String>,
    /// Skip font-style matching (~10-15% faster; translated text uses default
    /// styling instead of mimicking the original's bold/italic/weight).
    #[arg(long)]
    pub(crate) no_font_match: bool,
    /// Resume an interrupted run: reuse a stable project per `--folder`/`--khr`
    /// (instead of a fresh one each time) and skip pages already translated.
    #[arg(long)]
    pub(crate) resume: bool,
}
