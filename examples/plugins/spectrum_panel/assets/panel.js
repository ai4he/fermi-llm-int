/*
 * Example panel. Loaded by core/plugins.js after the main app.
 *
 * The panel registers itself, then refreshes whenever the app announces a
 * finished run. Core never imports this file; the manifest does.
 */
(function () {
    'use strict';
    const API = window.FermiLLM;
    if (!API) { return; }

    function render(host) {
        host.innerHTML = '<p class="example-panel-empty">No run yet.</p>';

        API.on('run:finished', function (result) {
            if (!result) { return; }
            const science = result.science_results || {};
            const rows = [
                ['Status', result.status || '—'],
                ['Target', science.target || '—'],
                ['TS', science.ts != null ? Number(science.ts).toFixed(1) : '—'],
                ['Flux', science.flux != null ? String(science.flux) : '—'],
            ];
            host.innerHTML =
                '<table class="example-panel-table">' +
                rows.map(function (r) {
                    return '<tr><th>' + r[0] + '</th><td>' + r[1] + '</td></tr>';
                }).join('') + '</table>';
        });
    }

    API.registerPanel({
        id: 'example-run-summary',
        title: 'Run summary (example plugin)',
        slot: 'results',
        render: render,
    });
})();
