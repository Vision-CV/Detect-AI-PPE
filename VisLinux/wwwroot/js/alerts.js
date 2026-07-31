﻿// alerts.js
console.log('alerts.js загружен');

// Дожидаемся DOM
document.addEventListener('DOMContentLoaded', function () {
    initAlertsPage();
});

async function initAlertsPage() {
    // Инициализация SignalR
    if (window.getSignalR) {
        try {
            const connection = await window.getSignalR();

            connection.on("OnNewIncident", (incident) => {
                console.log('🆕 Новый инцидент:', incident);
                addNewAlert(incident);
            });

            connection.on("IncidentUpdated", (incident) => {
                console.log('🔄 Обновление:', incident);
                updateIncidentInList(incident);
            });

            // НОВЫЙ ОБРАБОТЧИК ДЛЯ OnIncidentReviewed
            connection.on("OnIncidentReviewed", (incident) => {
                console.log('✅ Инцидент рассмотрен:', incident);
                updateIncidentInList(incident);
            });

            console.log('✅ SignalR обработчики для алертов зарегистрированы');
        } catch (e) {
            console.error('SignalR ошибка:', e);
        }
    }

    // Обновляем бейдж
    updateAlertBadge();
}

function getIconByPriority(priority) {
    const priorityKey = getPriorityClass(priority);
    if (priorityKey === 'critical') {
        return `<svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    } else if (priorityKey === 'warning') {
        return `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
    } else {
        return `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
    }
}

function renderAlertItem(incident) {
    const priorityClass = getPriorityClass(incident.priority);
    const statusText = getStatusText(incident.status);
    const statusClass = getStatusClass(incident.status);
    const incidentUrl = `/Dashboard/Incident?guid=${encodeURIComponent(incident.guid || incident.id || '')}`;
    const title = incident.title || incident.typeName || 'Инцидент';
    const desc = incident.description || 'Инцидент зафиксирован системой';

    return `
        <a href="${incidentUrl}">
            <div class="alert-item ${priorityClass}" data-type="${priorityClass}" data-id="${incident.guid || incident.id}">
                <div class="alert-icon ${priorityClass}">${getIconByPriority(incident.priority)}</div>
                <div class="alert-content">
                    <div class="alert-title">${escapeHtml(title)}</div>
                    <div class="alert-desc">${escapeHtml(desc)}</div>
                </div>
                <div class="alert-meta">
                    <div class="alert-time">${formatDateTime(incident.date || incident.timestamp)}</div>
                    <span class="status-badge ${statusClass}"><span class="status-dot"></span>${statusText}</span>
                </div>
            </div>
        </a>
    `;
}

function addNewAlert(incident) {
    const alertList = document.getElementById('alertList');
    if (!alertList) return;

    alertList.insertAdjacentHTML('afterbegin', renderAlertItem(incident));
    updateFilterCounts();
    updateAlertBadge();

    const priorityClass = getPriorityClass(incident.priority);
    const toastType = priorityClass === 'critical' ? 'error' : (priorityClass === 'warning' ? 'warning' : 'info');
    showToast('Новое событие', incident.title || incident.typeName || 'Инцидент', toastType);
}

function updateIncidentInList(incident) {
    const incidentId = incident.guid || incident.id;
    const link = document.querySelector(`.alert-item[data-id="${incidentId}"]`)?.closest('a');
    if (!link) return;

    const alertItem = link.querySelector('.alert-item');
    if (!alertItem) return;

    const statusBadge = alertItem.querySelector('.status-badge');
    if (statusBadge) {
        statusBadge.className = `status-badge ${getStatusClass(incident.status)}`;
        statusBadge.innerHTML = `<span class="status-dot"></span>${getStatusText(incident.status)}`;
    }

    const newPriorityClass = getPriorityClass(incident.priority);
    alertItem.classList.remove('critical', 'warning', 'info');
    alertItem.classList.add(newPriorityClass);
    alertItem.dataset.type = newPriorityClass;

    const iconDiv = alertItem.querySelector('.alert-icon');
    if (iconDiv) {
        iconDiv.innerHTML = getIconByPriority(incident.priority);
        iconDiv.classList.remove('critical', 'warning', 'info');
        iconDiv.classList.add(newPriorityClass);
    }

    updateFilterCounts();
    updateAlertBadge();
}

function updateFilterCounts() {
    const alerts = document.querySelectorAll('.alert-item');
    const criticals = document.querySelectorAll('.alert-item.critical').length;
    const warnings = document.querySelectorAll('.alert-item.warning').length;

    document.querySelectorAll('.filter-chip').forEach(chip => {
        const text = chip.innerText;
        if (text.includes('Критические')) {
            chip.innerHTML = chip.innerHTML.replace(/\d+/, criticals);
        } else if (text.includes('Предупреждения')) {
            chip.innerHTML = chip.innerHTML.replace(/\d+/, warnings);
        } else if (text.includes('Все')) {
            chip.innerHTML = chip.innerHTML.replace(/\d+/, alerts.length);
        }
    });
}

function updateAlertBadge() {
    const badge = document.querySelector('.nav-badge');
    if (!badge) return;

    const alerts = document.querySelectorAll('.alert-item');
    // ИЗМЕНЕНО: добавлена проверка на "Рассмотрено" и числовое значение 3
    const activeCount = Array.from(alerts).filter(alert => {
        const badge = alert.querySelector('.status-badge');
        const statusText = badge?.textContent || '';
        return !statusText.includes('Решено') && 
               !statusText.includes('Закрыто') &&
               !statusText.includes('Рассмотрено');
    }).length;

    badge.textContent = activeCount;
    badge.style.display = activeCount === 0 ? 'none' : 'inline-flex';
}

function filterAlerts(chip, type) {
    chip.parentElement?.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');

    document.querySelectorAll('.alert-item').forEach(alert => {
        alert.style.display = (type === 'all' || alert.dataset.type === type) ? 'flex' : 'none';
    });
}

async function markAllResolved() {
    const incidentIds = [];
    document.querySelectorAll('.alert-item').forEach(item => {
        const id = item.dataset.id;
        if (id) incidentIds.push(id);
    });

    if (incidentIds.length === 0) {
        showToast('Нет событий', 'Нет активных инцидентов', 'info');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/incidents/resolve-multiple`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ incidentIds })
        });

        if (response.ok) {
            const results = await response.json();
            results.forEach(inc => updateIncidentInList(inc));
            showToast('Готово', 'Все события обработаны', 'success');
        }
    } catch (e) {
        showToast('Ошибка', 'Проблема соединения', 'error');
    }
}