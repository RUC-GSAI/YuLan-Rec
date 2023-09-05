// 创建一个 WebSocket 连接
const userId = 123;  // 这只是一个示例用户ID，你可以根据需要更改它
const websocket = new WebSocket(`ws://81.70.235.253:18888/test`);

// 当 WebSocket 打开时触发的事件
websocket.onopen = function(event) {
    console.log("WebSocket is open now.", event);
};

// 当从服务器接收到消息时触发的事件
websocket.onmessage = function(event) {
    console.log("Received message:", event.data);
};

// 当 WebSocket 关闭时触发的事件
websocket.onclose = function(event) {
    console.log("WebSocket is closed now.", event);
};

// 当 WebSocket 出现错误时触发的事件
websocket.onerror = function(error) {
    console.error("WebSocket Error", error);
};
