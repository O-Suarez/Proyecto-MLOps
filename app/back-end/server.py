import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir la app FastAPI
app = FastAPI()

# Modelo de datos con Pydantic
class Data(BaseModel):
    NEds: int
    NActDays: int
    pagesWomen: int
    wikiprojWomen: int

# Ruta para recibir datos vía POST
@app.post("/api/parametros/")
async def receive_data(data: Data):
    logger.info(f"Datos recibidos: {data}")
    return {
        "message": "Datos recibidos correctamente",
        "data": data
    }

# Endpoint GET para verificar que la API está corriendo
@app.get("/api/")
async def get_status():
    return {
        "message": "La API está corriendo"
    }

# Configuración para ejecutar la app
if __name__ == "__main__":
    logger.info("Iniciando servidor...")
    uvicorn.run(
        app,  # Cambié "server:app" a app
        host="0.0.0.0",  # Cambié a "0.0.0.0" para que sea accesible desde fuera del contenedor
        port=8887,        
        log_level="info", # Nivel de logs
        reload=False       # Habilita la recarga automática en desarrollo
    )
