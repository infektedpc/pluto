
use crate::bindings::test_host::TestHost;
use crate::bindings::tests::{DOC, read_bool, read_num, read_str, run, test_tab};
use crate::host::{CanvasCall, DomHost, DomKey};
use crate::realm::RealmConfig;
use crate::tab::TabJs;

// ------------------------------------------------------------------
// Stage 15 (M7 Phase 3, Rec 8/9): 2D canvas bindings + tainting.
//
// These run page JS against the real `pluto-paint` rasteriser
// (via `TestHost`), so a passing test means the whole crossing
// works — bindings, host dispatch, and drawing — not just the
// Rust side.

/// A 2×2 canvas with a red 2D context, plus helpers to read
/// pixels back through `getImageData`.
const CANVAS_SETUP: &str = "const c = document.createElement('canvas'); \
     c.width = 4; c.height = 2; \
     document.body.appendChild(c); \
     const x = c.getContext('2d'); \
     globalThis.__px = (px, py) => { \
       const d = x.getImageData(px, py, 1, 1); \
       return [d.data[0], d.data[1], d.data[2], d.data[3]].join(','); \
     };";
#[test]
fn canvas_fill_rect_pixels_via_get_image_data() {
    let mut tab = test_tab();
    run(
        &mut tab,
        &format!(
            "{CANVAS_SETUP} \
             x.fillStyle = '#ff0000'; \
             x.fillRect(0, 0, 2, 2); \
             globalThis.__r = [x.getImageData(0, 0, 4, 2).width, x.getImageData(0, 0, 4, 2).height].join(','); \
             globalThis.__a = __px(0, 0); \
             globalThis.__b = __px(1, 0); \
             globalThis.__c = __px(2, 0); \
             globalThis.__d = __px(3, 1);"
        ),
    );
    assert_eq!(read_str(&mut tab, "__r"), "4,2");
    assert_eq!(read_str(&mut tab, "__a"), "255,0,0,255");
    assert_eq!(read_str(&mut tab, "__b"), "255,0,0,255");
    assert_eq!(read_str(&mut tab, "__c"), "0,0,0,0");
    assert_eq!(read_str(&mut tab, "__d"), "0,0,0,0");
}

#[test]
fn canvas_path_fill_triangle() {
    let mut tab = test_tab();
    run(
        &mut tab,
        &format!(
            "{CANVAS_SETUP} \
             x.fillStyle = 'rgb(0, 128, 0)'; \
             x.beginPath(); \
             x.moveTo(0, 0); x.lineTo(4, 0); x.lineTo(0, 2); x.closePath(); \
             x.fill(); \
             globalThis.__a = __px(1, 0); \
             globalThis.__b = __px(3, 1);"
        ),
    );
    assert_eq!(read_str(&mut tab, "__a"), "0,128,0,255");
    assert_eq!(read_str(&mut tab, "__b"), "0,0,0,0");
}

#[test]
fn canvas_gradient_fill_and_save_restore_and_transform() {
    let mut tab = test_tab();
    run(
        &mut tab,
        &format!(
            "{CANVAS_SETUP} \
             const g = x.createLinearGradient(0, 0, 4, 0); \
             g.addColorStop(0, '#ff0000'); \
             g.addColorStop(1, '#0000ff'); \
             x.fillStyle = g; \
             globalThis.__same = (x.fillStyle === g); \
             x.fillRect(0, 0, 4, 1); \
             globalThis.__left = __px(0, 0); \
             globalThis.__right = __px(3, 0); \
             x.save(); \
             x.fillStyle = '#00ff00'; \
             x.restore(); \
             x.fillRect(0, 1, 1, 1); \
             globalThis.__restored = __px(0, 1); \
             x.translate(3, 4); \
             const m = x.getTransform(); \
             globalThis.__m = [m.a, m.b, m.c, m.d, m.e, m.f].join(','); \
             x.lineWidth = 2.5; \
             globalThis.__lw = x.lineWidth;"
        ),
    );
    assert!(read_bool(&mut tab, "__same"));
    let left = read_str(&mut tab, "__left");
    let right = read_str(&mut tab, "__right");
    let l: Vec<u8> = left.split(',').map(|s| s.parse().unwrap()).collect();
    let r: Vec<u8> = right.split(',').map(|s| s.parse().unwrap()).collect();
    assert!(l[0] > 200 && l[2] < 60, "left end should be red, got {left}");
    assert!(r[2] > 200 && r[0] < 60, "right end should be blue, got {right}");
    // After restore the fill is the gradient again (not the green
    // set inside the save), so the bottom-left pixel is red-ish.
    let restored = read_str(&mut tab, "__restored");
    let rs: Vec<u8> = restored
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert!(rs[0] > 200, "restored fill should be red-ish, got {restored}");
    assert_eq!(read_str(&mut tab, "__m"), "1,0,0,1,3,4");
    assert_eq!(read_num(&mut tab, "__lw"), 2.5);
}

#[test]
fn canvas_draw_image_canvas_to_canvas_and_width_reset() {
    let mut tab = test_tab();
    run(
        &mut tab,
        &format!(
            "{CANVAS_SETUP} \
             const s = document.createElement('canvas'); \
             s.width = 2; s.height = 2; \
             const sx = s.getContext('2d'); \
             sx.fillStyle = '#ff0000'; \
             sx.fillRect(0, 0, 2, 2); \
             x.drawImage(s, 0, 0); \
             globalThis.__a = __px(1, 1); \
             x.clearRect(0, 0, 4, 2); \
             x.drawImage(s, 0, 0, 1, 1, 2, 0, 2, 2); \
             globalThis.__b = __px(3, 1); \
             globalThis.__c = __px(0, 0); \
             c.width = 4; \
             globalThis.__d = __px(1, 1);"
        ),
    );
    assert_eq!(read_str(&mut tab, "__a"), "255,0,0,255");
    assert_eq!(read_str(&mut tab, "__b"), "255,0,0,255");
    assert_eq!(read_str(&mut tab, "__c"), "0,0,0,0");
    // Setting `width` (even to the same value) resets the bitmap.
    assert_eq!(read_str(&mut tab, "__d"), "0,0,0,0");
}

#[test]
fn canvas_get_context_unknown_kind_null_and_same_object() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const c = document.createElement('canvas'); \
         globalThis.__null = (c.getContext('webgl') === null); \
         globalThis.__same = (c.getContext('2d') === c.getContext('2d')); \
         globalThis.__size = [c.width, c.height].join(','); \
         globalThis.__is_fn = (typeof c.getContext('2d').fillRect === 'function');",
    );
    assert!(read_bool(&mut tab, "__null"));
    assert!(read_bool(&mut tab, "__same"));
    assert_eq!(read_str(&mut tab, "__size"), "300,150");
    assert!(read_bool(&mut tab, "__is_fn"));
}

#[test]
fn canvas_measure_text_fill_text_and_to_data_url() {
    let mut tab = test_tab();
    run(
        &mut tab,
        &format!(
            "{CANVAS_SETUP} \
             globalThis.__w = x.measureText('hello').width; \
             x.font = '20px sans-serif'; \
             x.fillText('hi', 0, 1); \
             globalThis.__url = c.toDataURL().slice(0, 22);"
        ),
    );
    assert!(read_num(&mut tab, "__w") > 0.0);
    assert_eq!(read_str(&mut tab, "__url"), "data:image/png;base64,");
}

/// Build a tab whose document holds an `<img id="pic">` and
/// return the tab plus the image's key. The caller injects the
/// decoded pixels (via the host) and marks the image loaded.
fn canvas_taint_setup(origin_clean: bool) -> (TabJs, DomKey) {
    let mut host = TestHost::from_html(DOC);
    let img_key = host.dom.create_element("img");
    host.dom.set_attribute(img_key, "id", "pic");
    let body = host.body().unwrap();
    host.dom.append_child(body, img_key);
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(host));
    run(
        &mut tab,
        "const c = document.createElement('canvas'); \
         c.width = 2; c.height = 2; \
         document.body.appendChild(c); \
         const x = c.getContext('2d'); \
         const pic = document.getElementById('pic'); \
         globalThis.__draw = () => { x.drawImage(pic, 0, 0); }; \
         globalThis.__read = () => { \
           try { const d = x.getImageData(0, 0, 2, 2); return 'ok:' + d.data[0]; } \
           catch (e) { return 'err:' + e.name; } \
         }; \
         globalThis.__url = () => { \
           try { return 'ok:' + c.toDataURL().slice(0, 5); } \
           catch (e) { return 'err:' + e.name; } \
         }; \
         globalThis.__put = () => { \
           try { \
             x.putImageData({width: 1, height: 1, data: new Uint8ClampedArray([9, 9, 9, 255])}, 0, 0); \
             return 'ok'; \
           } catch (e) { return 'err:' + e.name; } \
         };",
    );
    tab.notify_image_loaded(img_key, 2, 2);
    let red = vec![255u8, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255];
    tab.dom_host()
        .unwrap()
        .borrow_mut()
        .canvas_call(
            img_key,
            CanvasCall::SetImagePixels {
                key: img_key,
                width: 2,
                height: 2,
                data: red,
                origin_clean,
            },
        );
    (tab, img_key)
}

#[test]
fn canvas_tainted_cross_origin_draw_image_throws_security_error() {
    let (mut tab, _) = canvas_taint_setup(false);
    run(&mut tab, "__draw();");
    assert_eq!(read_str(&mut tab, "__read()"), "err:SecurityError");
    assert_eq!(read_str(&mut tab, "__url()"), "err:SecurityError");
    // `putImageData` writes only and exposes nothing, so it stays
    // allowed on a tainted canvas.
    assert_eq!(read_str(&mut tab, "__put()"), "ok");
}

#[test]
fn canvas_clean_same_origin_draw_image_reads_pixels() {
    let (mut tab, _) = canvas_taint_setup(true);
    run(&mut tab, "__draw();");
    assert_eq!(read_str(&mut tab, "__read()"), "ok:255");
    assert_eq!(read_str(&mut tab, "__url()"), "ok:data:");
}
