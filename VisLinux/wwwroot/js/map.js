// map.js
console.log('map.js загружен');

document.addEventListener('DOMContentLoaded', function () {
    initMap();
    startMapDotsAnimation();
});

function initMap() {
    // Инициализация карты (если нужна)
    console.log('Карта инициализирована');
}

function zoomMap(dir) {
    const svg = document.querySelector('.map-svg');
    if (!svg) return;

    const current = parseFloat(svg.style.transform?.replace('scale(', '') || 1);
    const newScale = Math.max(0.5, Math.min(2, current + dir * 0.2));
    svg.style.transform = `scale(${newScale})`;
    svg.style.transformOrigin = 'center center';
    svg.style.transition = 'transform 0.3s ease';
}

function resetMap() {
    const svg = document.querySelector('.map-svg');
    if (svg) {
        svg.style.transform = 'scale(1)';
    }
    showToast('Карта сброшена', 'Масштаб восстановлен', 'success');
}

function startMapDotsAnimation() {
    setInterval(() => {
        document.querySelectorAll('.map-dot').forEach(dot => {
            const cx = parseFloat(dot.getAttribute('cx'));
            const cy = parseFloat(dot.getAttribute('cy'));
            const dx = (Math.random() - 0.5) * 2;
            const dy = (Math.random() - 0.5) * 2;
            dot.style.transition = 'all 2s ease';
            dot.setAttribute('cx', cx + dx);
            dot.setAttribute('cy', cy + dy);
        });
    }, 3000);
}

// Вызов из HTML (onclick атрибуты)
window.zoomMap = zoomMap;
window.resetMap = resetMap;