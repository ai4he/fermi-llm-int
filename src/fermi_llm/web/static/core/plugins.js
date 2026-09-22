/*
 * Frontend plugin loader.
 *
 * The page asks the server which UI extensions are active and injects what
 * each one declares. An extension is a manifest plus static assets:
 *
 *   {
 *     "name": "otherlab_spectrum_viewer",
 *     "label": "Spectrum viewer",
 *     "assets": ["/static/plugins/otherlab/viewer.js",
 *                "/static/plugins/otherlab/viewer.css"],
 *     "panels": [{"id": "spectrum", "title": "Spectrum", "slot": "results"}],
 *     "samples": [...]
 *   }
 *
 * Scripts run after the core app, so they can use window.FermiLLM. Nothing
 * here knows any extension by name: adding a panel never edits this file.
 */
(function () {
    'use strict';

    const API = window.FermiLLM = window.FermiLLM || {};

    // ---- extension points the core app exposes -------------------------
    API.panels = API.panels || [];
    API.hooks = API.hooks || {};

    /** Register a panel. `slot` is one of: results, sidebar, editor, footer. */
    API.registerPanel = function (panel) {
        if (!panel || !panel.id) { return; }
        API.panels.push(panel);
        mountPanel(panel);
    };

    /** Subscribe to a UI event ('session:loaded', 'run:finished', ...). */
    API.on = function (event, fn) {
        (API.hooks[event] = API.hooks[event] || []).push(fn);
    };

    /** Emit a UI event. Core calls this; plugins may too. */
    API.emit = function (event, payload) {
        (API.hooks[event] || []).forEach(function (fn) {
            try { fn(payload); } catch (err) {
                console.warn('[fermi-llm] hook failed for', event, err);
            }
        });
    };

    function slotElement(slot) {
        return document.querySelector('[data-plugin-slot="' + slot + '"]')
            || document.getElementById('results-content')
            || document.body;
    }

    function mountPanel(panel) {
        const host = slotElement(panel.slot || 'results');
        if (!host || document.getElementById('plugin-panel-' + panel.id)) { return; }
        const section = document.createElement('section');
        section.id = 'plugin-panel-' + panel.id;
        section.className = 'plugin-panel';
        if (panel.title) {
            const heading = document.createElement('h4');
            heading.textContent = panel.title;
            section.appendChild(heading);
        }
        const body = document.createElement('div');
        body.className = 'plugin-panel-body';
        section.appendChild(body);
        host.appendChild(section);
        if (typeof panel.render === 'function') {
            try { panel.render(body, API); } catch (err) {
                body.textContent = 'This panel failed to render.';
                console.warn('[fermi-llm] panel', panel.id, err);
            }
        }
    }

    function injectAsset(url) {
        return new Promise(function (resolve) {
            if (url.endsWith('.css')) {
                const link = document.createElement('link');
                link.rel = 'stylesheet';
                link.href = url;
                link.onload = link.onerror = resolve;
                document.head.appendChild(link);
                return;
            }
            const script = document.createElement('script');
            script.src = url;
            script.async = false;          // preserve declaration order
            script.onload = script.onerror = resolve;
            document.body.appendChild(script);
        });
    }

    async function load() {
        let extensions = [];
        try {
            const resp = await fetch('/api/ui/extensions');
            if (resp.ok) { extensions = (await resp.json()).extensions || []; }
        } catch (err) {
            console.warn('[fermi-llm] could not load UI extensions', err);
            return;                        // the core app works without them
        }
        for (const ext of extensions) {
            for (const asset of ext.assets || []) {
                await injectAsset(asset);
            }
            (ext.panels || []).forEach(function (panel) {
                if (!API.panels.some(function (p) { return p.id === panel.id; })) {
                    mountPanel(panel);
                }
            });
        }
        API.extensions = extensions;
        API.emit('plugins:loaded', extensions);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', load);
    } else {
        load();
    }
})();
