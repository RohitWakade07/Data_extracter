const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const statusSection = document.getElementById('statusSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');

// Drag and drop functionality
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.backgroundColor = '#f0f4ff';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#e0e0e0';
    uploadBox.style.backgroundColor = 'white';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#e0e0e0';
    uploadBox.style.backgroundColor = 'white';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
});

// File input change
fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
        uploadFile(file);
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Show status section
    hideAllSections();
    statusSection.style.display = 'flex';
    document.getElementById('statusText').textContent = `Processing ${file.name}...`;

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            displayError(data.error);
        }
    })
    .catch(error => {
        displayError(`Upload failed: ${error.message}`);
    });
}

function displayResults(data) {
    hideAllSections();
    resultsSection.style.display = 'flex';

    const safeSetText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };

    const results = data && data.results ? data.results : {};
    const entities = Array.isArray(results.entities) ? results.entities : [];
    const relationships = Array.isArray(results.relationships) ? results.relationships : [];

    safeSetText('resultFilename', data.filename || '');
    safeSetText('textPreview', data.text_preview || '');

    const entitiesCount = (typeof results.entities_count === 'number' && results.entities_count > 0)
        ? results.entities_count
        : entities.length;
    const relationshipsCount = (typeof results.relationships_count === 'number' && results.relationships_count > 0)
        ? results.relationships_count
        : relationships.length;

    safeSetText('entitiesCount', String(entitiesCount));
    safeSetText('relationshipsCount', String(relationshipsCount));
    safeSetText('vectorUUID', results.vector_document_uuid || 'N/A');
    safeSetText('workflowStatus', results.workflow_status || 'unknown');
    
    renderStructuredEntities(entities);
    renderStructuredRelationships(relationships);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

let currentGraphData = { entities: [], relationships: [] };

function renderStructuredEntities(entities) {
    const container = document.getElementById('entitiesStructured');
    if (!container) return;
    
    // Store for 3D graph
    currentGraphData.entities = entities;
    
    if (!entities || entities.length === 0) {
        container.innerHTML = '<p>No entities extracted</p>';
        return;
    }
    
    // Group entities by type
    const groupedByType = {};
    entities.forEach(entity => {
        const type = entity.type || 'unknown';
        if (!groupedByType[type]) {
            groupedByType[type] = [];
        }
        groupedByType[type].push(entity);
    });
    
    let html = '<div class="entities-list">';
    Object.keys(groupedByType).sort().forEach(type => {
        html += `<div class="entity-type-group">`;
        html += `<h4 class="entity-type">${type.charAt(0).toUpperCase() + type.slice(1)}</h4>`;
        html += `<ul class="entity-items">`;
        groupedByType[type].forEach(entity => {
            const confidence = entity.confidence ? (entity.confidence * 100).toFixed(0) : 100;
            html += `<li class="entity-item"><span class="entity-value">${escapeHtml(entity.value)}</span><span class="confidence-badge">${confidence}%</span></li>`;
        });
        html += `</ul>`;
        html += `</div>`;
    });
    html += '</div>';
    container.innerHTML = html;
}

function renderStructuredRelationships(relationships) {
    const container = document.getElementById('relationshipsStructured');
    if (!container) return;
    
    // Store for 3D graph
    currentGraphData.relationships = relationships;
    
    if (!relationships || relationships.length === 0) {
        container.innerHTML = '<p>No relationships extracted</p>';
        return;
    }
    
    let html = '<div class="relationships-table">';
    html += '<table>';
    html += '<thead><tr><th>From</th><th>Type</th><th>To</th><th>Confidence</th></tr></thead>';
    html += '<tbody>';
    
    relationships.forEach(rel => {
        const fromId = escapeHtml(rel.from_id || rel.from || 'Unknown');
        const toId = escapeHtml(rel.to_id || rel.to || 'Unknown');
        const relType = escapeHtml(rel.type || 'RELATED');
        const confidence = rel.confidence ? (rel.confidence * 100).toFixed(0) : 'N/A';
        
        html += `<tr>`;
        html += `<td class="rel-from">${fromId}</td>`;
        html += `<td class="rel-type">${relType}</td>`;
        html += `<td class="rel-to">${toId}</td>`;
        html += `<td class="rel-confidence">${confidence}%</td>`;
        html += `</tr>`;
    });
    
    html += '</tbody>';
    html += '</table>';
    html += '</div>';
    container.innerHTML = html;
}

function show3DGraph() {
    const modal = document.getElementById('graphModal');
    if (modal) {
        modal.style.display = 'flex';
        setTimeout(() => initializeGraph(currentGraphData), 100);
    }
}

function close3DGraph() {
    const modal = document.getElementById('graphModal');
    if (modal) {
        modal.style.display = 'none';
        const container = document.getElementById('graphContainer');
        if (container) container.innerHTML = '';
    }
}

function initializeGraph(data) {
    const container = document.getElementById('graphContainer');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Check if Three.js is available
    if (typeof THREE === 'undefined') {
        container.innerHTML = '<p style="color: #e74c3c;">Three.js library not loaded. Please refresh the page.</p>';
        return;
    }
    
    const width = container.clientWidth || 600;
    const height = 600;
    
    // Create wrapper for canvas and legend
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.gap = '20px';
    wrapper.style.width = '100%';
    wrapper.style.height = '600px';
    
    // Canvas container
    const canvasContainer = document.createElement('div');
    canvasContainer.style.flex = '1';
    canvasContainer.style.position = 'relative';
    
    // Legend container
    const legendContainer = document.createElement('div');
    legendContainer.style.width = '200px';
    legendContainer.style.overflowY = 'auto';
    legendContainer.style.padding = '15px';
    legendContainer.style.background = '#2a2a2a';
    legendContainer.style.borderRadius = '8px';
    legendContainer.style.color = 'white';
    legendContainer.style.fontSize = '0.85em';
    
    const legendHtml = `
        <h3 style="margin: 0 0 15px 0; color: #667eea;">Legend</h3>
        <div style="margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; font-size: 0.9em;">Entity Types:</h4>
            <div style="display: flex; flex-direction: column; gap: 6px;">
                <div style="display: flex; align-items: center; gap: 8px;"><span style="width: 12px; height: 12px; background: #ff6b6b; border-radius: 50%;"></span> Person</div>
                <div style="display: flex; align-items: center; gap: 8px;"><span style="width: 12px; height: 12px; background: #4ecdc4; border-radius: 50%;"></span> Organization</div>
                <div style="display: flex; align-items: center; gap: 8px;"><span style="width: 12px; height: 12px; background: #45b7d1; border-radius: 50%;"></span> Date</div>
                <div style="display: flex; align-items: center; gap: 8px;"><span style="width: 12px; height: 12px; background: #f9ca24; border-radius: 50%;"></span> Amount</div>
                <div style="display: flex; align-items: center; gap: 8px;"><span style="width: 12px; height: 12px; background: #6c5ce7; border-radius: 50%;"></span> Location</div>
            </div>
        </div>
        <div>
            <h4 style="margin: 0 0 8px 0; font-size: 0.9em;">Entities:</h4>
            <div id="entityList" style="display: flex; flex-direction: column; gap: 6px; max-height: 400px; overflow-y: auto;">
            </div>
        </div>
    `;
    
    legendContainer.innerHTML = legendHtml;
    wrapper.appendChild(canvasContainer);
    wrapper.appendChild(legendContainer);
    container.appendChild(wrapper);
    
    // Populate entity list in legend
    const entityList = legendContainer.querySelector('#entityList');
    const entityColors = {
        person: '#ff6b6b',
        organization: '#4ecdc4',
        date: '#45b7d1',
        amount: '#f9ca24',
        location: '#6c5ce7',
        unknown: '#95a5a6'
    };
    
    data.entities.forEach(entity => {
        const entityItem = document.createElement('div');
        entityItem.style.display = 'flex';
        entityItem.style.alignItems = 'center';
        entityItem.style.gap = '8px';
        entityItem.style.padding = '4px';
        entityItem.style.borderRadius = '4px';
        entityItem.style.backgroundColor = '#1a1a1a';
        
        const colorDot = document.createElement('span');
        colorDot.style.width = '8px';
        colorDot.style.height = '8px';
        colorDot.style.borderRadius = '50%';
        colorDot.style.background = entityColors[entity.type.toLowerCase()] || entityColors.unknown;
        colorDot.style.flexShrink = '0';
        
        const label = document.createElement('span');
        label.style.fontSize = '0.8em';
        label.style.overflow = 'hidden';
        label.style.textOverflow = 'ellipsis';
        label.style.whiteSpace = 'nowrap';
        label.textContent = entity.value.substring(0, 25);
        label.title = entity.value;
        
        entityItem.appendChild(colorDot);
        entityItem.appendChild(label);
        entityList.appendChild(entityItem);
    });
    
    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    
    const camera = new THREE.PerspectiveCamera(75, canvasContainer.clientWidth / height, 0.1, 10000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(canvasContainer.clientWidth, height);
    renderer.shadowMap.enabled = true;
    canvasContainer.appendChild(renderer.domElement);
    
    // Add lighting
    const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
    light1.position.set(100, 100, 100);
    light1.castShadow = true;
    scene.add(light1);
    
    const light2 = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(light2);
    
    // Create nodes (entities)
    const nodeMap = {};
    const nodeGroup = new THREE.Group();
    const labelGroup = new THREE.Group();
    
    const nodeEntityColors = {
        person: 0xff6b6b,
        organization: 0x4ecdc4,
        date: 0x45b7d1,
        amount: 0xf9ca24,
        location: 0x6c5ce7,
        unknown: 0x95a5a6
    };
    
    data.entities.forEach((entity, idx) => {
        const angle = (idx / data.entities.length) * Math.PI * 2;
        const radius = 80;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        const y = (Math.random() - 0.5) * 40;
        
        const color = nodeEntityColors[entity.type.toLowerCase()] || nodeEntityColors.unknown;
        
        const geometry = new THREE.IcosahedronGeometry(8, 2);
        const material = new THREE.MeshPhongMaterial({ color });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = { entity, label: entity.value.substring(0, 20) };
        
        nodeGroup.add(mesh);
        nodeMap[entity.value] = mesh;
        
        // Create text label for entity
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = 'Bold 20px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const label = entity.value.substring(0, 15);
        ctx.fillText(label, 128, 32);
        
        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture, sizeAttenuation: true });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(40, 10, 1);
        sprite.position.copy(mesh.position);
        sprite.position.y += 15;
        labelGroup.add(sprite);
    });
    
    scene.add(nodeGroup);
    scene.add(labelGroup);
    
    // Create edges (relationships)
    const lineGroup = new THREE.Group();
    const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x667eea, transparent: true, opacity: 0.8, linewidth: 2 });
    
    // Helper function to find entity node by matching ID or value
    const findEntityNode = (idOrValue) => {
        if (!idOrValue) return null;
        // Direct match by entity value
        if (nodeMap[idOrValue]) return nodeMap[idOrValue];
        // Try to find by stripping the type prefix (e.g., "person_John" -> match "John Smith")
        const parts = String(idOrValue).split('_');
        if (parts.length > 1) {
            const partialName = parts.slice(1).join(' ');
            for (const [value, node] of Object.entries(nodeMap)) {
                if (value.toLowerCase().includes(partialName.toLowerCase())) {
                    return node;
                }
            }
        }
        return null;
    };
    
    data.relationships.forEach(rel => {
        const fromNode = findEntityNode(rel.from_id || rel.from);
        const toNode = findEntityNode(rel.to_id || rel.to);
        
        if (fromNode && toNode) {
            // Create a tube geometry for better visibility
            const points = [fromNode.position.clone(), toNode.position.clone()];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, edgeMaterial);
            line.userData = { relationship: rel };
            lineGroup.add(line);
        }
    });
    
    scene.add(lineGroup);
    
    // Position camera
    camera.position.z = 200;
    
    // Mouse controls
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    
    renderer.domElement.addEventListener('mousedown', (e) => {
        isDragging = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
    });
    
    renderer.domElement.addEventListener('mousemove', (e) => {
        if (isDragging && e.buttons === 1) {
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;
            
            nodeGroup.rotation.y += deltaX * 0.005;
            nodeGroup.rotation.x += deltaY * 0.005;
            lineGroup.rotation.y += deltaX * 0.005;
            lineGroup.rotation.x += deltaY * 0.005;
            labelGroup.rotation.y += deltaX * 0.005;
            labelGroup.rotation.x += deltaY * 0.005;
        }
        previousMousePosition = { x: e.clientX, y: e.clientY };
    });
    
    renderer.domElement.addEventListener('mouseup', () => {
        isDragging = false;
    });
    
    renderer.domElement.addEventListener('wheel', (e) => {
        e.preventDefault();
        camera.position.z += e.deltaY * 0.1;
        camera.position.z = Math.max(50, Math.min(500, camera.position.z));
    });
    
    // Animation loop
    const animate = () => {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    };
    
    animate();
}

function displayError(error) {
    hideAllSections();
    errorSection.style.display = 'flex';
    document.getElementById('errorMessage').textContent = error;
}

function hideAllSections() {
    statusSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

function resetForm() {
    fileInput.value = '';
    hideAllSections();
    uploadBox.style.display = 'block';
    document.querySelector('.upload-section').scrollIntoView({ behavior: 'smooth' });
}

function toggleEntities() {
    const section = document.getElementById('entitiesSection');
    if (section) {
        section.style.display = section.style.display === 'none' ? 'block' : 'none';
    }
}

function toggleRelationships() {
    const section = document.getElementById('relationshipsSection');
    if (section) {
        section.style.display = section.style.display === 'none' ? 'block' : 'none';
    }
}

// Prevent default drag behavior on entire page
document.addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
});
