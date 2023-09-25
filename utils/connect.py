from fastapi import FastAPI, WebSocket
from typing import Dict, Optional
import asyncio
app = FastAPI()

class WebSocketManager:
    """
    Manage WebSocket connections.
    """
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, route:str,websocket: WebSocket):
        await websocket.accept()
        self.active_connections[route] = websocket
        return route

    def disconnect(self, route: str):
        if route in self.active_connections:
            del self.active_connections[route]

    def get_connection(self, route: str):
        return self.active_connections.get(route)
    
    async def send_personal_message(self, route: str,message: str):
        websocket = self.active_connections.get(route)
        if websocket:
            await websocket.send_text(message)

    async def send_and_wait_for_response(self, route: str, message: str) -> Optional[str]:
        websocket = self.active_connections.get(route)
        if not websocket:
            return None
        
        await websocket.send_text(message)
        response = await websocket.receive_text()
        input()
        return response
    
websocket_manager = WebSocketManager()
message_queue = []