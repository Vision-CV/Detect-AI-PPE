// incident.js
console.log('incident.js загружен');

document.addEventListener('DOMContentLoaded', function () {
    initIncidentPage();
});

async function initIncidentPage() {
    const incidentGuid = window.incidentGuid;
    if (!incidentGuid) return;

    // Отметка о просмотре (ИСПРАВЛЕНО)
    await markIncidentAsViewed(incidentGuid);

    // Навешиваем обработчики на кнопки
    const reportBtn = document.getElementById('reportBtn');
    const closeBtn = document.getElementById('closeBtn');

    if (reportBtn) {
        reportBtn.addEventListener('click', () => reportToAdmin());
    }

    if (closeBtn) {
        closeBtn.addEventListener('click', () => resolveCurrentIncident());
    }

    // Подписываемся на обновления этого инцидента через SignalR
    if (window.getSignalR) {
        try {
            const connection = await window.getSignalR();

            // НОВЫЙ ОБРАБОТЧИК ДЛЯ OnIncidentReviewed
            connection.on("OnIncidentReviewed", (reviewedIncident) => {
                if (reviewedIncident.guid === incidentGuid) {
                    console.log('✅ Инцидент рассмотрен:', reviewedIncident);
                    updateIncidentPage(reviewedIncident);
                }
            });

            console.log('✅ SignalR обработчики для инцидента зарегистрированы');
        } catch (e) {
            console.error('SignalR ошибка:', e);
        }
    }
}

// ИСПРАВЛЕНО: неправильный body
async function markIncidentAsViewed(incidentGuid) {
    try {
        const response = await fetch(`${API_BASE}/api/metrics/review-incident`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(incidentGuid)  // ✅ ИСПРАВЛЕНО: было { 'guid': `${incidentGuid}` }
        });

        if (!response.ok) {
            console.error('Ошибка отметки просмотра:', await response.text());
        } else {
            console.log('Просмотр зафиксирован для:', incidentGuid);
        }
    } catch (error) {
        console.error('Ошибка отметки просмотра:', error);
    }
}

// ИСПРАВЛЕНО: лишний слеш в URL и неправильная переменная в body
async function resolveCurrentIncident() {
    const incidentGuid = window.incidentGuid;
    const closeBtn = document.getElementById('closeBtn');

    if (!incidentGuid) return;

    try {
        if (closeBtn) {
            closeBtn.innerHTML = '<span class="spinner"></span> Обработка...';
            closeBtn.disabled = true;
        }

        // ИСПРАВЛЕНО: убран лишний слеш в конце URL
        const response = await fetch(`${API_BASE}/api/metrics/review-incident`, {  // ✅ убран /
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(incidentGuid)  // ✅ ИСПРАВЛЕНО: было JSON.stringify(guid) - переменная не определена
        });

        if (response.ok) {
            const result = await response.text();  // ✅ ИСПРАВЛЕНО: сервер возвращает текст, не JSON
            console.log('Успех:', result);
            showToast('Успех', 'Инцидент отмечен как рассмотренный', 'success');

            // Обновляем UI без перезагрузки
            const updatedIncident = { ...window.currentIncident, status: 3 };  // Reviewed = 3
            updateIncidentPage(updatedIncident);
        } else {
            const error = await response.text();
            showToast('Ошибка', `Не удалось отметить инцидент: ${error}`, 'error');
        }
    } catch (error) {
        console.error('Ошибка:', error);
        showToast('Ошибка', 'Проблема соединения с сервером', 'error');
    } finally {
        if (closeBtn && closeBtn.disabled) {
            closeBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><polyline points="22 4 12 14.01 9 11.01"/></svg> Закрыть инцидент';
            closeBtn.disabled = false;
        }
    }
}

function reportToAdmin() {
    // Отправка уведомления администратору зоны
    showToast('Уведомление отправлено', 'Администратор зоны оповещён', 'success');
    // TODO: реальный API вызов
}

// ИСПРАВЛЕНО: улучшена обработка статусов
function updateIncidentPage(incident) {
    // Сохраняем текущий инцидент для обновлений
    window.currentIncident = incident;

    // Обновляем статус
    const statusBadge = document.getElementById('statusBadge');
    if (statusBadge) {
        const statusClass = getStatusClass(incident.status);
        const statusText = getStatusText(incident.status);
        statusBadge.className = `incident-status-badge ${statusClass}`;
        statusBadge.textContent = statusText;
    }

    // Обновляем приоритет
    const priorityEl = document.getElementById('incidentPriorityLevel');
    if (priorityEl) {
        const priorityNames = { 0: 'Информационный', 1: 'Предупреждение', 2: 'Критический' };
        priorityEl.textContent = priorityNames[incident.priority] || incident.priority;

        // Обновляем цвет заголовка в зависимости от приоритета
        const titleEl = document.getElementById('incidentTitle');
        if (titleEl) {
            titleEl.classList.remove('critical-title', 'warning-title', 'info-title');
            if (incident.priority === 2 || incident.priority === 'Critical') {
                titleEl.classList.add('critical-title');
            } else if (incident.priority === 1 || incident.priority === 'Warning') {
                titleEl.classList.add('warning-title');
            }
        }
    }

    // ИСПРАВЛЕНО: правильная проверка статуса Reviewed
    const isResolved = incident.status === 1 ||      // Resolved
        incident.status === 'Resolved' ||
        incident.status === 3 ||      // Reviewed
        incident.status === 'Reviewed';

    const closeBtn = document.getElementById('closeBtn');
    if (closeBtn && isResolved) {
        closeBtn.disabled = true;
        closeBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><polyline points="22 4 12 14.01 9 11.01"/></svg> Закрыто';
    }
}

// Вспомогательные функции (добавьте, если их нет)
function getStatusClass(status) {
    const statusMap = {
        0: 'new',
        1: 'resolved',
        2: 'in-progress',
        3: 'reviewed'
    };
    return statusMap[status] || statusMap[0];
}

function getStatusText(status) {
    const textMap = {
        0: 'Новый',
        1: 'Решён',
        2: 'В работе',
        3: 'Рассмотрен'
    };
    return textMap[status] || 'Неизвестно';
}

function showToast(title, message, type) {
    // Ваша реализация toast-уведомлений
    console.log(`${title}: ${message}`);
    // Можно использовать простой alert для теста
    // alert(`${title}\n${message}`);
}