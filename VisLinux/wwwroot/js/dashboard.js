// dashboard.js
console.log('dashboard.js загружен');

// Глобальные переменные для уведомлений
let unreadCount = 0;
let notificationsList = [];

document.addEventListener('DOMContentLoaded', function () {
    initDashboard();
    window.addEventListener('resize', () => {
        const activeTab = document.querySelector('.tabs .tab.active');
        drawViolationsChart(activeTab?.innerText === 'Неделя' ? 'week' : 'month');
    });

    // Инициализация панели уведомлений
    initNotificationPanel();
    // Загрузка начальных уведомлений
    loadInitialNotifications();
});

// Глобальные функции для вызова из HTML
window.toggleNotifications = function () {
    const panel = document.getElementById('notifPanel');
    const overlay = document.getElementById('notifOverlay');

    if (!panel || !overlay) return;

    if (panel.classList.contains('open')) {
        closeNotifications();
    } else {
        openNotifications();
    }
};

window.clearNotifications = function () {
    notificationsList = [];
    unreadCount = 0;
    renderNotificationPanelList();
    updateNotificationBadges();
    if (typeof showToast === 'function') {
        showToast('Уведомления очищены', 'Все уведомления удалены', 'info');
    }
};

function closeNotifications() {
    const panel = document.getElementById('notifPanel');
    const overlay = document.getElementById('notifOverlay');

    if (panel && overlay) {
        panel.classList.remove('open');
        overlay.classList.remove('open');
    }
}

function openNotifications() {
    const panel = document.getElementById('notifPanel');
    const overlay = document.getElementById('notifOverlay');

    if (panel && overlay) {
        panel.classList.add('open');
        overlay.classList.add('open');
        // Сбрасываем счетчик непрочитанных при открытии
        unreadCount = 0;
        updateNotificationBadges();
    }
}

function initDashboard() {
    drawViolationsChart('week');
    drawDonutChart();
    drawSafetyIndex();
    updateMetrics();

    // SignalR для обновления графиков и уведомлений
    if (window.getSignalR) {
        window.getSignalR().then(connection => {
            connection.on("OnNewIncident", (incident) => {
                console.log('📊 Новый инцидент:', incident);
                refreshDashboardData(incident);
                // Показываем всплывающее уведомление
                showIncidentToast(incident);
                // Добавляем в список уведомлений
                addNotificationToList(incident);
            });
            connection.on("IncidentUpdated", (incident) => {
                console.log('🔄 Инцидент обновлён:', incident);
                refreshDashboardData(incident);
            });
        }).catch(e => console.error('SignalR:', e));
    }
}

function showIncidentToast(incident) {
    // Проверяем наличие функции showToast
    if (typeof showToast !== 'function') {
        console.log('showToast не доступен, создаем альтернативу');
        // Альтернативное уведомление через alert (временное решение)
        const priority = incident.priority === 2 || incident.priority === 'Critical' ? 'Критический' :
            (incident.priority === 1 || incident.priority === 'Warning' ? 'Предупреждение' : 'Информационный');
        alert(`${priority}: Новое событие`);
        return;
    }

    const priority = incident.priority === 2 || incident.priority === 'Critical' ? 'критический' :
        (incident.priority === 1 || incident.priority === 'Warning' ? 'предупреждение' : 'информационный');

    const typeNames = { 0: 'Без каски', 1: 'Опасная зона', 2: 'Курит', 3: 'Без жилета', 4: 'Прочее' };
    const typeName = typeNames[incident.type] || 'Нарушение';

    let title = 'Новое событие';
    let message = `${typeName} - ${incident.description}`;

    message = `Приоритет: ${priority}`;

    console.log('Показываем toast:', title, message, priority);
    showToast(title, message, priority);
}

function addNotificationToList(incident) {
    const priority = incident.priority === 2 || incident.priority === 'Critical' ? 'критический' :
        (incident.priority === 1 || incident.priority === 'Warning' ? 'предупреждение' : 'информационный');

    const typeNames = { 0: 'Без каски', 1: 'Опасная зона', 2: 'Курит', 3: 'Без жилета', 4: 'Прочее' };
    const typeName = typeNames[incident.type] || 'Нарушение';

    const notification = {
        id: incident.id || Date.now(),
        title: priority === 'critical' ? 'Критическое нарушение' : (priority === 'warning' ? 'Предупреждение' : 'Нарушение'),
        description: `${typeName}: ${incident.description || 'Зафиксировано нарушение'}`,
        time: new Date().toLocaleTimeString(),
        priority: priority,
        incident: incident
    };

    notificationsList.unshift(notification);
    // Ограничиваем количество уведомлений
    if (notificationsList.length > 50) notificationsList.pop();

    // Обновляем счетчик непрочитанных
    unreadCount++;
    updateNotificationBadges();

    // Обновляем список в панели уведомлений
    renderNotificationPanelList();
}

function loadInitialNotifications() {
    // Загружаем последние инциденты через fetch
    fetch('/api/incidents/recent?count=10')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(incidents => {
            if (incidents && incidents.length > 0) {
                incidents.forEach(incident => {
                    const priority = incident.priority === 2 || incident.priority === 'Critical' ? 'critical' :
                        (incident.priority === 1 || incident.priority === 'Warning' ? 'warning' : 'info');

                    const typeNames = { 0: 'Без каски', 1: 'Опасная зона', 2: 'Курит', 3: 'Без жилета', 4: 'Прочее' };
                    const typeName = typeNames[incident.type] || 'Нарушение';

                    notificationsList.push({
                        id: incident.id,
                        title: priority === 'critical' ? 'Критическое нарушение' : (priority === 'warning' ? 'Предупреждение' : 'Нарушение'),
                        description: `${typeName}: ${incident.description || 'Зафиксировано нарушение'}`,
                        time: new Date(incident.timestamp).toLocaleTimeString(),
                        priority: priority,
                        incident: incident
                    });
                });
                renderNotificationPanelList();
                // Не увеличиваем счетчик для начальных уведомлений
                unreadCount = 0;
                updateNotificationBadges();
            }
        })
        .catch(e => {
            console.log('Не удалось загрузить начальные уведомления:', e);
            // Добавляем тестовые уведомления для демонстрации
            addTestNotifications();
        });
}

function addTestNotifications() {
    // Тестовые уведомления для демонстрации работы панели
    notificationsList = [
    ];
    renderNotificationPanelList();
    unreadCount = 0;
    updateNotificationBadges();
}

function initNotificationPanel() {
    // Убеждаемся что обработчики настроены правильно
    const notifBtn = document.getElementById('notifBtn');
    if (notifBtn) {
        // Удаляем старые обработчики
        const newNotifBtn = notifBtn.cloneNode(true);
        notifBtn.parentNode.replaceChild(newNotifBtn, notifBtn);

        newNotifBtn.onclick = function (e) {
            e.preventDefault();
            e.stopPropagation();
            toggleNotifications();
        };
    }

    const notifOverlay = document.getElementById('notifOverlay');
    if (notifOverlay) {
        notifOverlay.onclick = function () {
            closeNotifications();
        };
    }

    // Закрытие по ESC
    document.removeEventListener('keydown', this._escHandler);
    this._escHandler = function (e) {
        if (e.key === 'Escape') {
            closeNotifications();
        }
    };
    document.addEventListener('keydown', this._escHandler);
}

function updateNotificationBadges() {
    // Обновляем точку на иконке колокольчика
    const notifBtn = document.getElementById('notifBtn');
    if (notifBtn) {
        const dot = notifBtn.querySelector('.notification-dot');
        if (dot) {
            if (unreadCount > 0) {
                dot.style.display = 'block';
            } else {
                dot.style.display = 'none';
            }
        }
    }

    // Обновляем бейдж в левом меню (вкладка События)
    const alertsBadge = document.getElementById('alertsBadge');
    if (alertsBadge) {
        if (unreadCount > 0) {
            alertsBadge.textContent = unreadCount > 99 ? '99+' : unreadCount;
            alertsBadge.style.display = 'inline-block';
        } else {
            alertsBadge.style.display = 'none';
        }
    }
}

function renderNotificationPanelList() {
    const alertList = document.getElementById('dynamicAlertList');
    if (!alertList) return;

    if (notificationsList.length === 0) {
        alertList.innerHTML = `
            <div style="text-align: center; padding: 40px 20px; color: #aeaeb2;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
                    <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
                </svg>
                <div style="margin-top: 12px; font-size: 14px;">Нет уведомлений</div>
            </div>
        `;
        return;
    }

    alertList.innerHTML = notificationsList.map(notif => `
        <div class="alert-item ${notif.priority}" onclick="goToAlertsPage()">
            <div class="alert-icon ${notif.priority}">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    ${notif.priority === 'critical' ?
            '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>' :
            (notif.priority === 'warning' ?
                '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>' :
                '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>')
        }
                </svg>
            </div>
            <div class="alert-content">
                <div class="alert-title">${escapeHtml(notif.title)}</div>
                <div class="alert-desc">${escapeHtml(notif.description)}</div>
            </div>
            <div class="alert-meta">
                <div class="alert-time">${escapeHtml(notif.time)}</div>
            </div>
        </div>
    `).join('');
}

// Глобальная функция для перехода на страницу событий
window.goToAlertsPage = function () {
    closeNotifications();
    window.location.href = '/Dashboard/Alerts';
};

// Функция просмотра инцидента (для обратной совместимости)
window.viewIncident = function (incidentId) {
    closeNotifications();
    window.location.href = `/Dashboard/Incident/${incidentId}`;
};

function refreshDashboardData(incident) {
    // Обновляем счётчик нарушений
    const violationsEl = document.getElementById('violationsCount');
    if (violationsEl) {
        const currentVal = parseInt(violationsEl.textContent) || 0;
        const newVal = currentVal + 1;
        if (typeof animateCounter === 'function') {
            animateCounter(violationsEl, newVal, 500);
        } else {
            violationsEl.textContent = newVal;
        }
        if (window.chartData) window.chartData.todayIncidents = newVal;
    }

    // Обновляем график
    const today = new Date();
    let dayIndex = today.getDay();
    dayIndex = dayIndex === 0 ? 6 : dayIndex - 1;
    const isCritical = incident.priority === 2 || incident.priority === 'Critical';

    if (window.chartData) {
        if (isCritical) {
            if (!window.chartData.weekCritical) window.chartData.weekCritical = [0, 0, 0, 0, 0, 0, 0];
            window.chartData.weekCritical[dayIndex] = (window.chartData.weekCritical[dayIndex] || 0) + 1;
        } else {
            if (!window.chartData.weekWarn) window.chartData.weekWarn = [0, 0, 0, 0, 0, 0, 0];
            window.chartData.weekWarn[dayIndex] = (window.chartData.weekWarn[dayIndex] || 0) + 1;
        }
    }

    const activeTab = document.querySelector('.tabs .tab.active');
    drawViolationsChart(activeTab?.innerText === 'Неделя' ? 'week' : 'month');

    // Обновляем круговую диаграмму
    const typeNames = { 0: 'Без каски', 1: 'Опасная зона', 2: 'Курит', 3: 'Без жилета', 4: 'Прочее' };
    const typeName = typeNames[incident.type] || 'Прочее';
    if (!window.chartData) window.chartData = {};
    if (!window.chartData.violationsByType) window.chartData.violationsByType = {};
    window.chartData.violationsByType[typeName] = (window.chartData.violationsByType[typeName] || 0) + 1;
    drawDonutChart();

    // Обновляем индекс безопасности
    let reduction = 0;
    if (incident.priority === 2 || incident.priority === 'Critical') reduction = 3;
    else if (incident.priority === 1 || incident.priority === 'Warning') reduction = 1;
    if (reduction > 0 && window.chartData) {
        const current = window.chartData.safetyIndex || 94;
        window.chartData.safetyIndex = Math.max(0, current - reduction);
        drawSafetyIndex();
        const safetyEl = document.getElementById('safetyIndexValue');
        if (safetyEl) {
            if (typeof animateCounter === 'function') {
                animateCounter(safetyEl, window.chartData.safetyIndex, 500);
            } else {
                safetyEl.textContent = window.chartData.safetyIndex;
            }
        }
    }
}

// Остальные функции (drawViolationsChart, drawDonutChart, drawSafetyIndex, updateMetrics, switchChartTab)
// остаются без изменений

function drawViolationsChart(period = 'week') {
    const canvas = document.getElementById('violationsChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.offsetWidth;
    const h = 280;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const padding = { top: 20, right: 20, bottom: 40, left: 40 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;
    ctx.clearRect(0, 0, w, h);

    let labels, data1, data2;
    if (period === 'week') {
        labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];
        data1 = window.chartData?.weekCritical || [2, 1, 3, 2, 4, 1, 0];
        data2 = window.chartData?.weekWarn || [3, 2, 2, 3, 2, 1, 1];
    } else {
        labels = ['Нед 1', 'Нед 2', 'Нед 3', 'Нед 4'];
        data1 = window.chartData?.monthCritical || [8, 12, 10, 7];
        data2 = window.chartData?.monthWarn || [10, 9, 11, 8];
    }

    const hasData = data1.some(v => v > 0) || data2.some(v => v > 0);
    const barGroupWidth = chartW / labels.length;
    const barWidth = barGroupWidth * 0.25;

    ctx.fillStyle = '#aeaeb2';
    ctx.font = '12px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    labels.forEach((label, i) => {
        ctx.fillText(label, padding.left + barGroupWidth * i + barGroupWidth / 2, h - 10);
    });

    if (!hasData) {
        ctx.fillStyle = '#aeaeb2';
        ctx.font = '14px -apple-system';
        ctx.fillText('Недостаточно данных', w / 2, h / 2);
        return;
    }

    const maxVal = Math.max(4, ...data1, ...data2);
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();
        ctx.fillStyle = '#aeaeb2';
        ctx.font = '11px -apple-system';
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(maxVal - (maxVal / 4) * i), padding.left - 8, y + 4);
    }

    data1.forEach((val, i) => {
        const x = padding.left + barGroupWidth * i + barGroupWidth / 2 - barWidth - 2;
        const barH = (val / maxVal) * chartH;
        if (barH > 0) {
            ctx.fillStyle = '#ff3b30';
            ctx.fillRect(x, padding.top + chartH - barH, barWidth, barH);
        }
    });

    data2.forEach((val, i) => {
        const x = padding.left + barGroupWidth * i + barGroupWidth / 2 + 2;
        const barH = (val / maxVal) * chartH;
        if (barH > 0) {
            ctx.fillStyle = '#ffd60a';
            ctx.fillRect(x, padding.top + chartH - barH, barWidth, barH);
        }
    });
}

function drawDonutChart() {
    const container = document.getElementById('donutChartContainer');
    if (!container) return;

    const data = window.chartData?.violationsByType || {
        "Без каски": 5, "Опасная зона": 4, "Без жилета": 3, "Прочее": 4
    };

    const colors = ['#ff3b30', '#ffd60a', '#0071e3', '#30d158', '#bf5af2'];
    const entries = Object.entries(data).filter(([, v]) => v > 0);
    const total = entries.reduce((s, [, v]) => s + v, 0);
    const radius = 48;
    const circumference = 2 * Math.PI * radius;

    let currentOffset = 0;
    let circles = '';
    entries.forEach(([, val], idx) => {
        const dash = (val / total) * circumference;
        circles += `<circle cx="60" cy="60" r="${radius}" fill="none" stroke="${colors[idx % colors.length]}" stroke-width="12" stroke-dasharray="${dash} ${circumference}" stroke-dashoffset="${-currentOffset}" stroke-linecap="round"/>`;
        currentOffset += dash;
    });

    let legend = '<div style="display:flex; flex-direction:column; gap:16px; width:100%; max-width:300px; margin-top:24px;">';
    entries.forEach(([name, val], idx) => {
        legend += `
            <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                <div style="display:flex; align-items:center; gap:8px; width:110px;">
                    <div style="width:10px; height:10px; border-radius:3px; background:${colors[idx % colors.length]}"></div>
                    <span style="font-size:13px; color:#1d1d1f;">${escapeHtml(name)}</span>
                </div>
                <div style="flex:1;"></div>
                <span style="min-width:40px; text-align:right; font-size:13px; font-weight:600; color:#1d1d1f;">${val} (${((val / total) * 100).toFixed(1)}%)</span>
            </div>
        `;
    });
    legend += '</div>';

    container.innerHTML = `
        <div style="display:flex; flex-direction:column; align-items:center;">
            <div style="position:relative; width:130px; height:130px;">
                <svg viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="${radius}" fill="none" stroke="#f0f0f3" stroke-width="8"/>
                    ${circles}
                </svg>
                <div style="position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                    <div style="font-size:32px; font-weight:700; color:#1d1d1f;">${total}</div>
                    <div style="font-size:11px; color:#6e6e73;">всего</div>
                </div>
            </div>
            ${legend}
        </div>
    `;
}

function drawSafetyIndex() {
    const container = document.getElementById('safetyIndexContainer');
    if (!container) return;

    const safetyIndex = window.chartData?.safetyIndex || 94;
    const components = window.chartData?.safetyComponents || {
        "Средства защиты": 96, "Зона доступа": 93, "Оборудование": 91, "Поведение": 87
    };

    const radius = 50;
    const percent = safetyIndex / 100;
    const dash = percent * 2 * Math.PI * radius;
    const color = safetyIndex < 70 ? '#ff3b30' : (safetyIndex < 85 ? '#ffd60a' : '#30d158');

    container.innerHTML = `
        <div style="display:flex; flex-direction:column; align-items:center; gap:24px;">
            <div style="position:relative; width:130px; height:130px;">
                <svg viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="${radius}" fill="none" stroke="#f0f0f3" stroke-width="8"/>
                    <circle cx="60" cy="60" r="${radius}" fill="none" stroke="${color}" stroke-width="8" stroke-dasharray="${dash} ${2 * Math.PI * radius}" stroke-linecap="round"/>
                </svg>
                <div style="position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                    <div style="font-size:32px; font-weight:700; color:${color}">${safetyIndex}</div>
                    <div style="font-size:11px; color:#6e6e73">из 100</div>
                </div>
            </div>
            <div style="display:flex; flex-direction:column; gap:16px; width:100%; max-width:300px;">
                ${Object.entries(components).map(([name, value]) => {
        const barColor = value < 70 ? '#ff3b30' : (value < 85 ? '#ffd60a' : '#30d158');
        return `
                        <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                            <span style="width:110px; font-size:13px;">${escapeHtml(name)}</span>
                            <div style="flex:1;"><div style="height:6px; background:#f0f0f3; border-radius:3px;"><div style="width:${value}%; height:100%; background:${barColor}; border-radius:3px;"></div></div></div>
                            <span style="min-width:40px; text-align:right; font-size:13px; font-weight:600;">${value}%</span>
                        </div>
                    `;
    }).join('')}
            </div>
        </div>
    `;
}

function updateMetrics() {
    const violationsEl = document.getElementById('violationsCount');
    if (violationsEl && window.chartData?.todayIncidents) {
        animateCounter(violationsEl, window.chartData.todayIncidents);
    }
}

function switchChartTab(el, period) {
    el.parentElement?.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    drawViolationsChart(period);
}