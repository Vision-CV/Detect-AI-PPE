// signalr.js
let connection = null;

window.getSignalR = async () => {
    if (connection?.state === "Connected") return connection;
    if (connection) await connection.stop();

    connection = new signalR.HubConnectionBuilder()
        .withUrl("/signalHub")
        .build();

    await connection.start();
    return connection;
};

window.addEventListener('beforeunload', () => connection?.stop());