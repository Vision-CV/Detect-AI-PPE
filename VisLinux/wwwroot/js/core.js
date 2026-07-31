// core.js
console.log('core.js загружен');

const API_BASE = window.apiBaseUrl || '';

function showToast(title, message, type = 'info') {
    console.log('showToast вызван:', title, message, type);

    const container = document.getElementById('toastContainer');
    if (!container) {
        console.error('toastContainer не найден');
        return;
    }

    const icons = {
        success: '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#30d158" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        error: '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#ff3b30" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        warning: '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#ffd60a" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        info: '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#0071e3" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        critical: '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#ff3b30" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    };

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <div class="toast-icon">${icons[type] || icons.info}</div>
        <div class="toast-content">
            <div class="toast-title">${escapeHtml(title)}</div>
            <div class="toast-message">${escapeHtml(message)}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.classList.add('removing');setTimeout(()=>this.parentElement.remove(),300)">
            <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
    `;
    container.appendChild(toast);
    setTimeout(() => {
        if (toast.parentElement) {
            toast.classList.add('removing');
            setTimeout(() => toast.remove(), 300);
        }
    }, 4000);
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>]/g, function (m) {
        if (m === '&') return '&amp;';
        if (m === '<') return '&lt;';
        if (m === '>') return '&gt;';
        return m;
    });
}

function animateCounter(el, target, duration = 1500) {
    let start = 0;
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(start + (target - start) * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function getPriorityClass(priority) {
    const map = { 0: 'info', 1: 'warning', 2: 'critical', 'Standard': 'info', 'Warning': 'warning', 'Critical': 'critical' };
    return map[priority] || 'info';
}

// ИЗМЕНЕНО: добавлен статус "reviewed" для рассмотренных инцидентов
function getStatusClass(status) {
    const map = {
        0: 'pending',
        1: 'online',
        2: 'offline',
        3: 'reviewed',
        'Pending': 'pending',
        'Resolved': 'online',
        'Closed': 'offline',
        'Reviewed': 'reviewed'
    };
    return map[status] || 'pending';
}

// ИЗМЕНЕНО: добавлен текст "Рассмотрено" для статуса 3
function getStatusText(status) {
    const map = {
        0: 'В обработке',
        1: 'Решено',
        2: 'Закрыто',
        3: 'Рассмотрено',
        'Pending': 'В обработке',
        'Resolved': 'Решено',
        'Closed': 'Закрыто',
        'Reviewed': 'Рассмотрено'
    };
    return map[status] || 'В обработке';
}

function formatDateTime(dateStr) {
    if (!dateStr) return '---';
    const date = new Date(dateStr);
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
}