//! Import on-disk image files as pages in a session.
//!
//! Shared by the headless CLI `translate` command (folder input). Mirrors the
//! server-side dance in `koharu-rpc`'s `POST /pages/from-paths`: decode →
//! `blobs.put_bytes` → emit one `Op::AddPage` per image.

use anyhow::{Context, Result};
use camino::Utf8Path;
use image::GenericImageView;
use koharu_core::{ImageData, ImageRole, Node, NodeId, NodeKind, Op, Page, PageId, Transform};

use crate::session::ProjectSession;

/// File extensions we treat as importable images (compared lowercased).
const IMAGE_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff", "avif",
];

/// Whether `path` looks like an image we can import (by extension).
pub fn is_image_path(path: &Utf8Path) -> bool {
    path.extension()
        .map(|ext| IMAGE_EXTS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Natural-order sort by file name so `page-2.png` precedes `page-10.png`.
fn sort_natural(paths: &mut [&Utf8Path]) {
    paths.sort_by(|a, b| {
        let an = a.file_name().unwrap_or_else(|| a.as_str());
        let bn = b.file_name().unwrap_or_else(|| b.as_str());
        natord::compare(an, bn)
    });
}

/// Decode each image, store its bytes as a blob, and append one page per image
/// (natural-sorted by file name). Pages are appended after any existing pages.
/// Returns the new page ids in order.
pub fn import_image_files<P: AsRef<Utf8Path>>(
    session: &ProjectSession,
    paths: &[P],
) -> Result<Vec<PageId>> {
    let mut paths: Vec<&Utf8Path> = paths.iter().map(AsRef::as_ref).collect();
    sort_natural(&mut paths);

    let starting_index = session.scene.read().pages.len();
    let mut ops = Vec::with_capacity(paths.len());
    let mut ids = Vec::with_capacity(paths.len());
    for (i, path) in paths.iter().enumerate() {
        let bytes =
            std::fs::read(path.as_std_path()).with_context(|| format!("read image {path}"))?;
        let img =
            image::load_from_memory(&bytes).with_context(|| format!("decode image {path}"))?;
        let (w, h) = img.dimensions();
        let blob = session.blobs.put_bytes(&bytes)?;
        let filename = path.file_name().unwrap_or("page").to_string();

        let mut page = Page::new(&filename, w, h);
        ids.push(page.id);
        let node_id = NodeId::new();
        page.nodes.insert(
            node_id,
            Node {
                id: node_id,
                transform: Transform::default(),
                visible: true,
                kind: NodeKind::Image(ImageData {
                    role: ImageRole::Source,
                    blob,
                    opacity: 1.0,
                    natural_width: w,
                    natural_height: h,
                    name: Some(filename),
                }),
            },
        );
        ops.push(Op::AddPage {
            page,
            at: starting_index + i,
        });
    }

    if !ops.is_empty() {
        session.apply(Op::Batch {
            ops,
            label: "Import images".into(),
        })?;
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_sort_orders_numbered_pages() {
        let p1 = Utf8Path::new("a/page-1.png");
        let p2 = Utf8Path::new("a/page-2.png");
        let p10 = Utf8Path::new("a/page-10.png");
        let mut v = vec![p10, p2, p1];
        sort_natural(&mut v);
        assert_eq!(v, vec![p1, p2, p10]);
    }

    #[test]
    fn image_extensions_detected_case_insensitively() {
        assert!(is_image_path(Utf8Path::new("x.PNG")));
        assert!(is_image_path(Utf8Path::new("x.jpeg")));
        assert!(!is_image_path(Utf8Path::new("x.txt")));
        assert!(!is_image_path(Utf8Path::new("noext")));
    }
}
