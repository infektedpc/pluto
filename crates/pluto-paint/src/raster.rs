//! CPU triangle rasteriser (Stage 15, M7 Phase 2, Rec 5).
//!
//! The worker-side rasteriser for vector content the GPU path
//! cannot express (radial gradients, multi-stop linear gradients)
//! and for CSS radial-gradient backgrounds. It reuses the
//! tessellator from [`crate::path`]: lyon flattens the path to
//! triangles, and this module fills them into an RGBA bitmap with
//! barycentric colour interpolation (so a linear gradient is exact
//! per pixel, and a solid fill is a constant colour).
//!
//! No antialiasing in this phase (hard edges — documented): the
//! vector pipeline is the crisp route for solid shapes, and this
//! rasteriser is the fallback for gradient shapes. The bitmap is
//! produced at the box's used size (fixed raster resolution —
//! re-rastered on resize, when the display list rebuilds) and
//! full-frame only (no partial updates for live canvas yet).

use crate::path::VectorVertex;
use pluto_css::{Gradient, GradientKind};
use crate::DecodedImage;

/// Visit every covered pixel centre of every triangle, in
/// order, with the triangle and its barycentric weights. The
/// top-left rule decides coverage (a pixel exactly on a
/// top-or-left edge is in, on the other edges out — so shared
/// edges are drawn once, not twice); the caller decides what to
/// do with the pixel. Vertices are in bitmap pixels.
fn for_each_covered_pixel<F>(vertices: &[VectorVertex], width: u32, height: u32, mut f: F)
where
    F: FnMut(usize, usize, &[VectorVertex], f32, f32, f32),
{
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let cross = |ax: f32, ay: f32, bx: f32, by: f32| ax * by - ay * bx;
    for tri in vertices.chunks_exact(3) {
        let (a, mut b, mut c) = (tri[0], tri[1], tri[2]);
        // Normalise winding so the signed area is positive; the
        // edge functions below then have the interior on their
        // positive side.
        let area2 = cross(
            b.position[0] - a.position[0],
            b.position[1] - a.position[1],
            c.position[0] - a.position[0],
            c.position[1] - a.position[1],
        );
        if area2 == 0.0 {
            continue;
        }
        if area2 < 0.0 {
            std::mem::swap(&mut b, &mut c);
        }
        let area2 = area2.abs();
        // Pixel bbox, clamped to the bitmap.
        let min_x = a
            .position[0]
            .min(b.position[0])
            .min(c.position[0])
            .floor()
            .max(0.0) as usize;
        let max_x = a
            .position[0]
            .max(b.position[0])
            .max(c.position[0])
            .ceil()
            .min(w as f32) as usize;
        let min_y = a
            .position[1]
            .min(b.position[1])
            .min(c.position[1])
            .floor()
            .max(0.0) as usize;
        let max_y = a
            .position[1]
            .max(b.position[1])
            .max(c.position[1])
            .ceil()
            .min(h as f32) as usize;
        if min_x >= max_x || min_y >= max_y {
            continue;
        }
        // Edge functions F_ab(p) = cross(b-a, p-a): positive
        // inside (after the winding normalisation). An edge is
        // top-or-left when it goes right (horizontal) or up
        // (non-horizontal) — pixels exactly on such an edge are
        // included, on the other edges excluded (top-left rule).
        let edge = |p: [f32; 2], q: [f32; 2], r: [f32; 2]| -> (f32, bool) {
            let dx = q[0] - p[0];
            let dy = q[1] - p[1];
            let f = cross(dx, dy, r[0] - p[0], r[1] - p[1]);
            let top_left = (dy == 0.0 && dx > 0.0) || dy < 0.0;
            (f, top_left)
        };
        for y in min_y..max_y {
            for x in min_x..max_x {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let (fab, tl_ab) = edge(a.position, b.position, p);
                let (fbc, tl_bc) = edge(b.position, c.position, p);
                let (fca, tl_ca) = edge(c.position, a.position, p);
                let inside = |f: f32, tl: bool| f > 0.0 || (f == 0.0 && tl);
                if !(inside(fab, tl_ab) && inside(fbc, tl_bc) && inside(fca, tl_ca)) {
                    continue;
                }
                // Barycentric weights (each edge function over
                // the opposite vertex's value, which is area2).
                let wa = fbc / area2;
                let wb = fca / area2;
                let wc = fab / area2;
                f(x, y, tri, wa, wb, wc);
            }
        }
    }
}

/// Rasterise coloured triangles into an RGBA8 bitmap, `width` ×
/// `height`, top-left origin, y down. Vertices are in bitmap
/// pixels. Straight alpha, source-over compositing onto
/// transparent black. Pure.
pub fn rasterize_triangles(vertices: &[VectorVertex], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut rgba = vec![0u8; w * h * 4];
    rasterize_triangles_into(&mut rgba, vertices, width, height);
    rgba
}

/// Like [`rasterize_triangles`], but composites into an existing
/// bitmap (so successive shapes paint over earlier ones in
/// document order). `rgba` must be `width * height * 4` bytes.
pub fn rasterize_triangles_into(
    rgba: &mut [u8],
    vertices: &[VectorVertex],
    width: u32,
    height: u32,
) {
    let w = width as usize;
    for_each_covered_pixel(vertices, width, height, |x, y, tri, wa, wb, wc| {
        let mut srgb = [0.0f32; 4];
        for i in 0..4 {
            srgb[i] = wa * tri[0].color[i] + wb * tri[1].color[i] + wc * tri[2].color[i];
        }
        let idx = (y * w + x) * 4;
        let sa = srgb[3].clamp(0.0, 1.0);
        if sa <= 0.0 {
            return;
        }
        let da = rgba[idx + 3] as f32 / 255.0;
        let out_a = sa + da * (1.0 - sa);
        if out_a <= 0.0 {
            return;
        }
        for i in 0..3 {
            let sc = srgb[i].clamp(0.0, 1.0);
            let dc = rgba[idx + i] as f32 / 255.0;
            let out = (sc * sa + dc * da * (1.0 - sa)) / out_a;
            rgba[idx + i] = (out.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        rgba[idx + 3] = (out_a.clamp(0.0, 1.0) * 255.0).round() as u8;
    });
}

/// Rasterise triangles to a 1-byte-per-pixel coverage mask (255
/// where a pixel centre is inside a triangle, 0 elsewhere).
/// Used for gradient-filled shapes, whose colour is evaluated
/// per pixel rather than interpolated from vertices (a radial
/// gradient's centre is never a vertex). Pure.
pub fn rasterize_coverage(vertices: &[VectorVertex], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut mask = vec![0u8; w * h];
    for_each_covered_pixel(vertices, width, height, |x, y, _, _, _, _| {
        mask[y * w + x] = 255;
    });
    mask
}

/// Rasterise a CSS gradient over a `width` × `height` rect into
/// an RGBA bitmap (one pixel per gradient evaluation — exact for
/// linear and radial alike). Used for CSS radial-gradient
/// backgrounds (linear backgrounds go through the vector quad in
/// [`crate::path::linear_gradient_quad`]). Pure.
pub fn rasterize_gradient_rect(gradient: &Gradient, width: u32, height: u32) -> DecodedImage {
    let w = width as usize;
    let h = height as usize;
    let mut rgba = vec![0u8; w * h * 4];
    let rect = (0.0, 0.0, width as f32, height as f32);
    let is_linear = matches!(gradient.kind, GradientKind::Linear { .. });
    for y in 0..h {
        for x in 0..w {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let t = if is_linear {
                gradient.linear_t(p, rect)
            } else {
                gradient.radial_t(p, rect)
            };
            let c = gradient.color_at(t);
            let idx = (y * w + x) * 4;
            rgba[idx] = (c[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[idx + 1] = (c[1].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[idx + 2] = (c[2].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[idx + 3] = (c[3].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    DecodedImage {
        width,
        height,
        rgba,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::VectorVertex;

    fn v(x: f32, y: f32, color: [f32; 4]) -> VectorVertex {
        VectorVertex {
            position: [x, y],
            color,
        }
    }

    #[test]
    fn triangle_fill_covers_its_interior() {
        let red = [1.0, 0.0, 0.0, 1.0];
        // A right triangle covering the top-left half of a 4x4
        // bitmap (vertices (0,0), (4,0), (0,4)).
        let tris = vec![v(0.0, 0.0, red), v(4.0, 0.0, red), v(0.0, 4.0, red)];
        let rgba = rasterize_triangles(&tris, 4, 4);
        let px = |x: usize, y: usize| rgba[(y * 4 + x) * 4];
        // Inside: top-left corner and the diagonal's inner side.
        assert_eq!(px(0, 0), 255);
        assert_eq!(px(1, 1), 255);
        assert_eq!(px(0, 2), 255);
        // On the diagonal (a right/bottom edge under the top-left
        // rule) the pixel centre is excluded; outside is empty.
        assert_eq!(px(0, 3), 0);
        assert_eq!(px(3, 3), 0);
        assert_eq!(px(3, 2), 0);
    }

    #[test]
    fn shared_edge_is_drawn_once_not_twice() {
        // Two triangles sharing the diagonal of a 2x2 bitmap,
        // both 50% red: the shared edge must not double-blend
        // (a pixel on the edge is covered by exactly one
        // triangle under the top-left rule).
        let half = [1.0, 0.0, 0.0, 0.5];
        let tris = vec![
            v(0.0, 0.0, half),
            v(2.0, 0.0, half),
            v(0.0, 2.0, half),
            v(2.0, 0.0, half),
            v(2.0, 2.0, half),
            v(0.0, 2.0, half),
        ];
        let rgba = rasterize_triangles(&tris, 2, 2);
        for i in 0..4 {
            // Every pixel is exactly one 50%-red layer over
            // transparent black: alpha 128, red 255 (straight
            // alpha stores the source colour, not premultiplied).
            assert_eq!(rgba[i * 4 + 3], 128, "pixel {i} alpha");
            assert_eq!(rgba[i * 4], 255, "pixel {i} red");
        }
    }

    #[test]
    fn barycentric_interpolation_blends_vertex_colours() {
        // A triangle with red/green/blue corners: the centre
        // pixel must be a blend of all three (not one corner's
        // colour).
        let tris = vec![
            v(0.0, 0.0, [1.0, 0.0, 0.0, 1.0]),
            v(9.0, 0.0, [0.0, 1.0, 0.0, 1.0]),
            v(0.0, 9.0, [0.0, 0.0, 1.0, 1.0]),
        ];
        let rgba = rasterize_triangles(&tris, 9, 9);
        // Pixel (3, 3) is strictly inside (its centre (3.5, 3.5)
        // sits off the hypotenuse x + y = 9, which the top-left
        // rule excludes).
        let idx = (3 * 9 + 3) * 4;
        let (r, g, b) = (rgba[idx], rgba[idx + 1], rgba[idx + 2]);
        assert!(r > 0 && g > 0 && b > 0, "centre is a blend: {r} {g} {b}");
        assert_eq!(rgba[idx + 3], 255);
    }

    #[test]
    fn gradient_rect_evaluates_linear_and_radial() {
        use pluto_css::{Gradient as CssGradient, GradientKind, GradientStop, RadialExtent, RadialShape};
        let linear = CssGradient {
            kind: GradientKind::Linear { angle_deg: 90.0 },
            stops: vec![
                GradientStop { offset: 0.0, color: [255, 0, 0, 255] },
                GradientStop { offset: 1.0, color: [0, 0, 255, 255] },
            ],
        };
        let img = rasterize_gradient_rect(&linear, 10, 1);
        assert_eq!(img.width, 10);
        assert_eq!(img.height, 1);
        // Leftmost pixel is mostly red, rightmost mostly blue.
        assert!(img.rgba[0] > 200 && img.rgba[2] < 55);
        let last = 9 * 4;
        assert!(img.rgba[last + 2] > 200 && img.rgba[last] < 55);
        let radial = CssGradient {
            kind: GradientKind::Radial {
                center: (0.5, 0.5),
                shape: RadialShape::Circle,
                extent: RadialExtent::ClosestSide,
            },
            stops: vec![
                GradientStop { offset: 0.0, color: [255, 255, 255, 255] },
                GradientStop { offset: 1.0, color: [0, 0, 0, 255] },
            ],
        };
        let img = rasterize_gradient_rect(&radial, 9, 9);
        let centre = (4 * 9 + 4) * 4;
        assert!(img.rgba[centre] > 200, "radial centre is white");
        assert!(img.rgba[0] < 55, "radial corner is black");
    }
}
