# Bundled nav-bar icons

The three toolbar icons (Back, Forward, Reload), as SVGs compiled into
the Pluto binary with `include_bytes!`.

## Why bundled

`cosmic::widget::icon::from_name` only searches the Cosmic icon theme.
On a machine without Cosmic icons installed (e.g. KDE/Breeze), the
lookup finds nothing and the button silently draws blank — no error,
no placeholder. Bundling the SVGs makes the icons render wherever
Pluto runs, and `.symbolic(true)` tints them to the theme's icon
color (white on the dark nav bar).

## Source

`go-previous-symbolic.svg`, `go-next-symbolic.svg`,
`view-refresh-symbolic.svg` — from
[pop-os/cosmic-icons](https://github.com/pop-os/cosmic-icons)
(`freedesktop/scalable/actions/`), licensed
[CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Unmodified.

## Nav-feature icons (hand-drawn, same style)

`star-symbolic.svg` (outline star), `star-filled-symbolic.svg`
(filled star), `open-menu-symbolic.svg` (hamburger) — drawn for
Pluto in the same 16x16, single `#232323` fill style so the
symbolic tint paints them like the Cosmic icons. The outline star
is a filled even-odd path (outer star + inner star cut out), not
a stroke, so the tint applies to it the same way.
