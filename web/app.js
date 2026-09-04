// Viewport for the exact-contact packer.
//
// Every number on this page came from the bridge, which got it from the solver.
// Nothing here computes a density, a verdict, or a placement. The browser draws
// what it is given and sends back what the user did.
//
// Axes. The solver is Z-up in millimetres. three.js is Y-up. The mapping is
//     three (x, y, z) = solver (x, z, y)
// applied once, in toThree() and fromThree(), and nowhere else.

(function () {
    'use strict';

    var CHECK_DELAY_MS = 50;
    var COLOR_ITEM = 0x8a9bb0;
    var COLOR_SELECTED = 0x0b6b53;
    var COLOR_WARN = 0xb3261e;
    var COLOR_POINTS = 0x5b6068;
    var COLOR_BOX = 0x1c1f23;

    var S = {
        scene: null, camera: null, renderer: null, raycaster: null,
        data: null, meshes: {}, points: null, boxLine: null,
        target: new THREE.Vector3(), spherical: { r: 500, theta: 0.8, phi: 1.05 },
        orbit: null, pan: null, drag: null, lastCheck: 0, lastCheckResult: null
    };

    function toThree(p) { return new THREE.Vector3(p[0], p[2], p[1]); }
    function fromThree(v) { return [v.x, v.z, v.y]; }
    function $(id) { return document.getElementById(id); }
    function fmt(x, d) { return (x === null || x === undefined || isNaN(x)) ? 'n/a' : Number(x).toFixed(d); }

    function api(path, body) {
        var opts = body === undefined ? { method: 'GET' }
            : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
        return fetch(path, opts).then(function (r) { return r.json(); });
    }

    // ---------------------------------------------------------------- scene

    function init() {
        var vp = $('viewport');
        S.scene = new THREE.Scene();
        S.scene.background = new THREE.Color(0xe8eaec);
        S.camera = new THREE.PerspectiveCamera(45, 1, 1, 5000);
        S.renderer = new THREE.WebGLRenderer({ antialias: true });
        vp.appendChild(S.renderer.domElement);
        S.raycaster = new THREE.Raycaster();

        S.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
        var sun = new THREE.DirectionalLight(0xffffff, 0.55);
        sun.position.set(300, 500, 200);
        S.scene.add(sun);

        resize();
        window.addEventListener('resize', resize);
        bindPointer();
        bindUi();
        loadScene();
        animate();
    }

    function resize() {
        var vp = $('viewport');
        var w = vp.clientWidth || Math.floor(window.innerWidth * 0.65);
        var h = vp.clientHeight || (window.innerHeight - 40);
        S.renderer.setSize(w, h, false);
        S.camera.aspect = w / h;
        S.camera.updateProjectionMatrix();
    }

    function updateCamera() {
        var s = S.spherical;
        S.camera.position.set(
            S.target.x + s.r * Math.sin(s.phi) * Math.cos(s.theta),
            S.target.y + s.r * Math.cos(s.phi),
            S.target.z + s.r * Math.sin(s.phi) * Math.sin(s.theta));
        S.camera.lookAt(S.target);
    }

    function clearScene() {
        Object.keys(S.meshes).forEach(function (k) { S.scene.remove(S.meshes[k]); });
        S.meshes = {};
        if (S.points) { S.scene.remove(S.points); S.points = null; }
        if (S.boxLine) { S.scene.remove(S.boxLine); S.boxLine = null; }
        if (S.labels) { S.scene.remove(S.labels); S.labels = null; }
    }

    // Clear air between the displaced capture cloud and the container box.
    var SCAN_GAP_MM = 60;

    // figures/scene_common.py draws bodies at PT_SIZE 3.4 and the table at 0.5,
    // thinned four to one. These are the same proportions in world millimetres.
    var BODY_PT_MM = 3.4;
    var TABLE_PT_MM = 1.2;
    var TABLE_STEP = 4;
    // A few pixels of a LiDAR colour reads washed out on screen, so each channel
    // is pushed away from the point's own mean. This is DISPLAY only. The solver
    // never reads a colour, and the stored capture is untouched.
    var SAT = 1.55;

    function punchColour(r, g, b, out, i) {
        var m = (r + g + b) / 3;
        out[i] = Math.min(1, Math.max(0, (m + (r - m) * SAT) / 255));
        out[i + 1] = Math.min(1, Math.max(0, (m + (g - m) * SAT) / 255));
        out[i + 2] = Math.min(1, Math.max(0, (m + (b - m) * SAT) / 255));
    }

    // A name floating above each body. Sprites, so the text always faces the
    // camera and never has to be re-oriented as the view orbits. DISPLAY only.
    var LABEL_LIFT_MM = 26;

    function makeLabel(text, worldHeightMM) {
        var pad = 8, font = 34;
        var mc = document.createElement('canvas');
        var mx = mc.getContext('2d');
        mx.font = font + 'px ui-monospace, Consolas, monospace';
        var w = Math.ceil(mx.measureText(text).width) + pad * 2;
        var h = font + pad * 2;
        mc.width = w; mc.height = h;
        var cx = mc.getContext('2d');
        cx.font = font + 'px ui-monospace, Consolas, monospace';
        cx.textBaseline = 'middle';
        cx.fillStyle = 'rgba(255,255,255,0.82)';
        cx.fillRect(0, 0, w, h);
        cx.strokeStyle = 'rgba(0,0,0,0.28)';
        cx.lineWidth = 2;
        cx.strokeRect(1, 1, w - 2, h - 2);
        cx.fillStyle = '#1a1a1a';
        cx.fillText(text, pad, h / 2);
        var tex = new THREE.CanvasTexture(mc);
        if (THREE.SRGBColorSpace) { tex.colorSpace = THREE.SRGBColorSpace; }
        tex.minFilter = THREE.LinearFilter;
        var sp = new THREE.Sprite(new THREE.SpriteMaterial({
            map: tex, transparent: true, depthTest: false
        }));
        sp.renderOrder = 10;
        sp.scale.set(worldHeightMM * (w / h), worldHeightMM, 1);
        return sp;
    }

    function addScanLabels(d) {
        if (!d.scan_anchors) { return; }
        Object.keys(d.scan_anchors).forEach(function (label) {
            var a = d.scan_anchors[label];
            var sp = makeLabel(label, 15);
            // Solver z is up, three.js y is up, and the cloud is displaced.
            sp.position.set(a.xy_mm[0], a.top_mm + LABEL_LIFT_MM,
                            a.xy_mm[1] + (S.scanOffset || 0));
            S.labels.add(sp);
        });
    }

    function addPackedLabels(d) {
        // The label rests on the crown of its own body, so the name and the
        // object it names are never in doubt. depthTest is off on the sprite,
        // so a label stays readable even when a nearer body would cover it.
        // Tall bodies are added last and so draw over short ones.
        var pl = (d.placements || []).slice().sort(function (a, b) {
            return (a.centre_mm[2] + a.axes_mm[2]) - (b.centre_mm[2] + b.axes_mm[2]);
        });
        pl.forEach(function (q) {
            var sp = makeLabel(q.label, 10);
            sp.position.set(q.centre_mm[0],
                            q.centre_mm[2] + q.axes_mm[2] + 6,
                            q.centre_mm[1]);
            S.labels.add(sp);
        });
    }

    function buildScene(d) {
        clearScene();
        var b = d.box_mm;                                   // [width x, depth y, height z]
        var geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(b[0], b[2], b[1]));
        S.boxLine = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: COLOR_BOX }));
        S.boxLine.position.set(b[0] / 2, b[2] / 2, b[1] / 2);
        S.scene.add(S.boxLine);
        S.labels = new THREE.Group();
        S.scene.add(S.labels);

        if (d.points_mm && d.points_mm.length) {
            // The cloud is the CAPTURE, in table coordinates. The box and the
            // ellipsoids are the PLAN. They share only an origin, so drawing
            // both untranslated ran the table straight through the box. The
            // cloud is shifted clear along +y for DISPLAY ONLY. Nothing the
            // solver reads is touched, and no reported number changes.
            var maxY = -Infinity, minY = Infinity;
            for (var s0 = 0; s0 < d.points_mm.length; s0++) {
                var yv = d.points_mm[s0][1];
                if (yv > maxY) { maxY = yv; }
                if (yv < minY) { minY = yv; }
            }
            S.scanOffset = (b[1] - minY) + SCAN_GAP_MM;
            // The cloud arrives ordered table first, bodies after. Drawing the
            // two halves separately is what lets the produce read solid while
            // the table stays a faint ground, exactly as the paper figure does.
            var nTable = d.n_table_points || 0;
            var haveRGB = d.points_rgb && d.points_rgb.length === d.points_mm.length;

            function makeCloud(from, to, step, sizeMM, fallbackColour) {
                var idx = [];
                for (var k = from; k < to; k += step) { idx.push(k); }
                if (!idx.length) { return null; }
                var pos = new Float32Array(idx.length * 3);
                var col = haveRGB ? new Float32Array(idx.length * 3) : null;
                for (var j = 0; j < idx.length; j++) {
                    var q = d.points_mm[idx[j]];
                    pos[3 * j] = q[0];
                    pos[3 * j + 1] = q[2];
                    pos[3 * j + 2] = q[1] + S.scanOffset;
                    if (col) {
                        var c = d.points_rgb[idx[j]];
                        punchColour(c[0], c[1], c[2], col, 3 * j);
                    }
                }
                var geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
                var mat;
                if (col) {
                    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
                    mat = new THREE.PointsMaterial({ vertexColors: true, size: sizeMM });
                } else {
                    mat = new THREE.PointsMaterial({ color: fallbackColour, size: sizeMM });
                }
                return new THREE.Points(geo, mat);
            }

            var tableCloud = makeCloud(0, nTable, TABLE_STEP, TABLE_PT_MM, COLOR_POINTS);
            var bodyCloud = makeCloud(nTable, d.points_mm.length, 1, BODY_PT_MM, COLOR_POINTS);
            S.points = new THREE.Group();
            if (tableCloud) {
                tableCloud.material.opacity = 0.55;
                tableCloud.material.transparent = true;
                S.points.add(tableCloud);
            }
            if (bodyCloud) { S.points.add(bodyCloud); }
            S.scene.add(S.points);
            addScanLabels(d);
        }

        d.placements.forEach(function (pl) {
            var m = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16),
                new THREE.MeshLambertMaterial({ color: COLOR_ITEM }));
            var a = pl.axes_mm;                              // axes[2] is vertical once placed
            m.scale.set(a[0], a[2], a[1]);
            m.rotation.y = -pl.yaw;
            m.position.copy(toThree(pl.centre_mm));
            m.userData = { label: pl.label, home: m.position.clone(), fragile: pl.fragile };
            S.meshes[pl.label] = m;
            S.scene.add(m);
        });
        addPackedLabels(d);

        S.target.set(b[0] / 2, b[2] / 3, b[1] / 2);
        S.target.z += (S.scanOffset || 0) * 0.22;
        S.spherical.r = Math.max(b[0], b[1], b[2]) * 2.2 + (S.scanOffset || 0) * 0.55;
        updateCamera();
    }

    function animate() {
        requestAnimationFrame(animate);
        S.renderer.render(S.scene, S.camera);
    }

    // ---------------------------------------------------------------- data

    function loadScene() {
        return api('/api/scene').then(applyScene).catch(function (e) { setStatus('bridge unreachable'); });
    }

    function applyScene(d) {
        if (!d || d.error && !d.placements) { setStatus('error'); return; }
        S.data = d;
        buildScene(d);
        fillBoxSelector(d);
        fillItemStrip(d);
        fillItemCounts(d);
        updateMeasurements(d);
        setStatus(d.air_ok ? 'ok' : 'no key');
    }

    function fillBoxSelector(d) {
        var sel = $('box-selector');
        sel.innerHTML = '';
        d.box_catalog.forEach(function (b, i) {
            var o = document.createElement('option');
            o.value = String(i);
            var nm = (d.box_names && d.box_names[i]) ? d.box_names[i] + '  ' : '';
            // The optimised box comes back as a measured float. One decimal is
            // the honest resolution for a container dimension, and the raw
            // double is unreadable in a dropdown.
            function mm(v) { return (Math.round(v * 10) / 10).toFixed(1); }
            o.textContent = nm + mm(b[0]) + ' x ' + mm(b[1]) + ' x ' + mm(b[2]) + ' mm';
            if (i === d.box_index) o.selected = true;
            sel.appendChild(o);
        });
    }

    // One cell per library type across the top of the 3D view: the name, the
    // mass, and the count handed to the solver. The point cloud lives in the
    // main viewport, it is not repeated here.
    function fillItemStrip(d) {
        var host = $('item-strip');
        if (!host || !d.library) { return; }
        host.innerHTML = '';
        d.library.forEach(function (it) {
            var cell = document.createElement('div');
            cell.className = 'item-cell';
            var text = document.createElement('div');
            text.className = 'cell-text';
            var nm = document.createElement('span');
            nm.className = 'cell-name'; nm.textContent = it.label;
            var sub = document.createElement('span');
            sub.className = 'cell-sub';
            sub.textContent = it.mass_g > 0 ? (it.mass_g.toFixed(0) + ' g')
                                            : 'not weighed';
            text.appendChild(nm); text.appendChild(sub);
            var box = document.createElement('input');
            box.type = 'number'; box.min = '0';
            box.max = String(d.max_copies || 12);
            box.value = String((d.counts && d.counts[it.label] != null) ? d.counts[it.label] : 1);
            box.id = 'count-' + it.label;
            box.addEventListener('change', applyCounts);
            cell.appendChild(text); cell.appendChild(box);
            host.appendChild(cell);
        });
    }

    // One spinner per library type. The count is what the SOLVER is given, so
    // three eggs are three separate bodies, not one drawn three times.
    function fillItemCounts(d) {
        var host = $('item-counts');
        if (host && d.library) { host.innerHTML = ''; }
        if (!host || !d.library) { host = null; }
        if (host) { d.library.forEach(function (it) {
            var row = document.createElement('div');
            row.className = 'count-row';
            var name = document.createElement('span');
            var grams = it.mass_g > 0 ? (' ' + it.mass_g.toFixed(0) + ' g') : ' not weighed';
            name.textContent = it.label + grams;
            var box = document.createElement('input');
            box.type = 'number'; box.min = '0';
            box.max = String(d.max_copies || 12);
            box.value = String((d.counts && d.counts[it.label] != null) ? d.counts[it.label] : 1);
            box.id = 'count-' + it.label;
            row.appendChild(name); row.appendChild(box);
            host.appendChild(row);
        }); }
        var n = $('n-bodies'); if (n) { n.textContent = String((d.items || []).length); }
        var tm = $('total-mass');
        if (tm) { tm.textContent = d.masses_known ? (d.total_mass_g.toFixed(0) + ' g') : 'not weighed'; }
        var pm = $('packed-mass');
        if (pm) { pm.textContent = d.masses_known ? (d.packed_mass_g.toFixed(0) + ' g') : 'not weighed'; }
    }

    // Ask the SOLVER for the smallest free-size box that takes every item. The
    // dimensions come back from real packs, never from a volume estimate.
    function optimizeBox() {
        setStatus('optimizing');
        appendTranscript('bridge: searching for the smallest box, this takes about 25 s');
        api('/api/optimize', {})
            .then(function (d) {
                if (!d || d.error) { appendTranscript('bridge: ' + ((d && d.error) || 'failed')); setStatus('error'); return; }
                applyScene(d);
                var o = d.optimize || {};
                if (!o.found) {
                    appendTranscript('bridge: no box in the search placed every item');
                    return;
                }
                var b = o.box_mm;
                appendTranscript('bridge: optimized box ' + b[0] + ' x ' + b[1] + ' x ' + b[2]
                    + ' mm, ' + o.volume_L + ' L, gate ' + d.gate);
                if (o.beats_catalog) {
                    appendTranscript('bridge: smallest stock box that would hold it is ' + o.beats_catalog);
                }
            })
            .catch(function () { setStatus('error'); });
    }

    function applyCounts() {
        if (!S.data || !S.data.library) { return; }
        var counts = {};
        S.data.library.forEach(function (it) {
            var el = $('count-' + it.label);
            counts[it.label] = el ? parseInt(el.value, 10) : 1;
        });
        setStatus('packing');
        api('/api/counts', { counts: counts })
            .then(function (d) {
                if (d && d.error) { appendTranscript('bridge: ' + d.error); setStatus('error'); return; }
                applyScene(d);
                appendTranscript('bridge: re-packed ' + (d.items || []).length + ' bodies, gate ' + d.gate);
                // An OPT box sized for a different item set will strand items
                // and the count change is the reason, not the packer.
                if (d.opt_stale) {
                    appendTranscript('bridge: WARNING the OPT box was sized for a different item set, '
                        + 'press Optimized Box again or pick a stock size');
                }
            })
            .catch(function () { setStatus('error'); });
    }

    function updateMeasurements(d) {
        var r = d.report || {};
        $('container-density').textContent = fmt(r.container, 3);
        $('hull-density').textContent = fmt(r.hull, 3);
        $('laguerre-density').textContent = (r.laguerre === null || r.laguerre === undefined)
            ? 'unavailable (no interior cell)' : fmt(r.laguerre, 3);
        $('fill-height').textContent = fmt(r.fill_height_mm, 1) + ' mm';
        setGate(d.gate, d.max_penetration_mm, d.tolerance_mm, null);

        var bm = d.benchmark || {};
        $('best-q').textContent = fmt(bm.best_q, 2);
        $('wall-ratio').textContent = fmt(bm.wall_ratio, 1);
        $('ceiling').textContent = (bm.equal_sphere_ceiling === null || bm.equal_sphere_ceiling === undefined)
            ? 'n/a (n=' + d.placements.length + ')' : fmt(bm.equal_sphere_ceiling, 4);

        var extra = [];
        if (d.unplaced && d.unplaced.length) extra.push('unplaced: ' + d.unplaced.join(', '));
        if (d.error) extra.push(d.error);
        if (extra.length) appendTranscript('bridge: ' + extra.join(' | '));
    }

    function setGate(gate, pen, tol, against) {
        var el = $('gate-status');
        var ok = gate === 'PASSED' || gate === 'OK';
        var txt = (ok ? 'PASSED' : (gate || 'n/a')) + ' ' + fmt(pen, 4) + ' mm';
        if (!ok && against) txt += ' against ' + against;
        if (tol !== undefined && tol !== null) txt += '   tolerance ' + fmt(tol, 3) + ' mm';
        el.textContent = txt;
        el.style.color = ok ? 'var(--accent)' : 'var(--warn)';
    }

    function setStatus(s) {
        var el = $('air-status');
        el.textContent = '[ASU AIR: ' + s + ']';
        el.style.color = s === 'ok' ? 'var(--accent)' : 'var(--warn)';
    }

    // ---------------------------------------------------------------- transcript

    function appendTranscript(line) {
        var t = $('transcript');
        t.textContent += line + '\n';
        t.scrollTop = t.scrollHeight;
    }

    function renderTranscript(entries) {
        entries.forEach(function (e) {
            if (e.role === 'model' && e.name) {
                appendTranscript('model -> ' + e.name + '(' + JSON.stringify(e.arguments || {}) + ')');
            } else if (e.role === 'tool') {
                appendTranscript('tool  <- ' + JSON.stringify(e.content));
            } else if (e.content) {
                appendTranscript((e.role || 'system') + ' : ' + e.content);
            }
        });
    }

    function sendRules() {
        var input = $('rules-input');
        var text = input.value.trim();
        if (!text) return;
        input.value = '';
        appendTranscript('user  : ' + text);
        setStatus('working');
        api('/api/turn', { text: text }).then(function (out) {
            if (out.error) { appendTranscript('bridge: ' + out.error); setStatus('error'); return; }
            renderTranscript(out.transcript || []);
            if (out.final_text) appendTranscript('model : ' + out.final_text);
            if (out.scene) applyScene(out.scene);
            setStatus('ok');
        }).catch(function () { setStatus('error'); });
    }

    function rePack() {
        setStatus('packing');
        api('/api/pack', { box_index: parseInt($('box-selector').value, 10) })
            .then(function (d) { applyScene(d); appendTranscript('bridge: re-packed, gate ' + d.gate); })
            .catch(function () { setStatus('error'); });
    }

    // ---------------------------------------------------------------- pointer

    function ndc(ev) {
        var r = S.renderer.domElement.getBoundingClientRect();
        return new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1,
                                 -((ev.clientY - r.top) / r.height) * 2 + 1);
    }

    function pickItem(ev) {
        S.raycaster.setFromCamera(ndc(ev), S.camera);
        var hits = S.raycaster.intersectObjects(Object.keys(S.meshes).map(function (k) { return S.meshes[k]; }));
        return hits.length ? hits[0].object : null;
    }

    function pointOnPlane(ev, y) {
        S.raycaster.setFromCamera(ndc(ev), S.camera);
        var plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -y);
        var out = new THREE.Vector3();
        return S.raycaster.ray.intersectPlane(plane, out) ? out : null;
    }

    function bindPointer() {
        var c = S.renderer.domElement;
        c.addEventListener('pointerdown', function (ev) {
            c.setPointerCapture(ev.pointerId);
            if (ev.button === 0 && !ev.shiftKey) {
                var m = pickItem(ev);
                if (m) {
                    var hit = pointOnPlane(ev, m.position.y);
                    S.drag = { mesh: m, offset: hit ? m.position.clone().sub(hit) : new THREE.Vector3() };
                    m.material.color.setHex(COLOR_SELECTED);
                    S.lastCheckResult = { ok: true };
                    return;
                }
            }
            if (ev.button === 1 || ev.shiftKey) S.pan = { x: ev.clientX, y: ev.clientY };
            else S.orbit = { x: ev.clientX, y: ev.clientY };
        });

        c.addEventListener('pointermove', function (ev) {
            if (S.drag) {
                var hit = pointOnPlane(ev, S.drag.mesh.position.y);
                if (!hit) return;
                S.drag.mesh.position.copy(hit.add(S.drag.offset));
                var now = performance.now();
                if (now - S.lastCheck >= CHECK_DELAY_MS) {
                    S.lastCheck = now;
                    var m = S.drag.mesh;
                    api('/api/check', { label: m.userData.label, centre_mm: fromThree(m.position) })
                        .then(function (res) {
                            if (!S.drag || S.drag.mesh !== m) return;
                            S.lastCheckResult = res;
                            if (res.ok) {
                                m.material.color.setHex(COLOR_SELECTED);
                                setGate('OK', res.penetration_mm, res.tolerance_mm, null);
                            } else {
                                m.material.color.setHex(COLOR_WARN);
                                setGate(res.outside_box ? 'OUTSIDE BOX' : 'OVERLAP REJECTED',
                                        res.penetration_mm, res.tolerance_mm, res.against);
                            }
                        });
                }
                return;
            }
            if (S.orbit) {
                S.spherical.theta -= (ev.clientX - S.orbit.x) * 0.006;
                S.spherical.phi = Math.min(1.5, Math.max(0.1, S.spherical.phi - (ev.clientY - S.orbit.y) * 0.006));
                S.orbit = { x: ev.clientX, y: ev.clientY };
                updateCamera();
            } else if (S.pan) {
                var k = S.spherical.r * 0.0015;
                var right = new THREE.Vector3().crossVectors(S.camera.getWorldDirection(new THREE.Vector3()), S.camera.up).normalize();
                S.target.addScaledVector(right, -(ev.clientX - S.pan.x) * k);
                S.target.y += (ev.clientY - S.pan.y) * k;
                S.pan = { x: ev.clientX, y: ev.clientY };
                updateCamera();
            }
        });

        function release() {
            if (S.drag) {
                var m = S.drag.mesh;
                var res = S.lastCheckResult;
                if (!res || !res.ok) {
                    m.position.copy(m.userData.home);            // snap back to the last valid pose
                    setGate(S.data ? S.data.gate : 'n/a', S.data ? S.data.max_penetration_mm : null,
                            S.data ? S.data.tolerance_mm : null, null);
                } else {
                    m.userData.home.copy(m.position);
                }
                m.material.color.setHex(COLOR_ITEM);
                S.drag = null;
            }
            S.orbit = null;
            S.pan = null;
        }
        c.addEventListener('pointerup', release);
        c.addEventListener('pointercancel', release);
        c.addEventListener('wheel', function (ev) {
            ev.preventDefault();
            S.spherical.r = Math.min(4000, Math.max(50, S.spherical.r * (ev.deltaY > 0 ? 1.1 : 0.9)));
            updateCamera();
        }, { passive: false });
    }

    // ---------------------------------------------------------------- ui

    function bindUi() {
        $('reset-view').addEventListener('click', function () {
            S.spherical.theta = 0.8; S.spherical.phi = 1.05;
            if (S.data) {
                var b = S.data.box_mm;
                S.target.set(b[0] / 2, b[2] / 3, b[1] / 2 + (S.scanOffset || 0) * 0.22);
                S.spherical.r = Math.max(b[0], b[1], b[2]) * 2.2 + (S.scanOffset || 0) * 0.55;
            }
            updateCamera();
        });
        $('re-pack').addEventListener('click', rePack);
        $('box-selector').addEventListener('change', rePack);
        $('optimize-box').addEventListener('click', optimizeBox);
        $('apply-counts').addEventListener('click', applyCounts);
        $('send-rules').addEventListener('click', sendRules);
        $('rules-input').addEventListener('keypress', function (e) { if (e.key === 'Enter') sendRules(); });
    }

    window.addEventListener('load', init);
})();
