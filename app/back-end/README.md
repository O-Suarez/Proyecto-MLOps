# Documentación de la API Predictive

Esta documentación describe cómo construir y ejecutar la API Predictive utilizando FastAPI y Docker.

## Tabla de Contenidos
- [Construcción de la Imagen Docker](#construcción-de-la-imagen-docker)
- [Ejecución del Contenedor Docker](#ejecución-del-contenedor-docker)
- [Endpoints de la API](#endpoints-de-la-api)
  - [POST /api/parametros/](#post-apiparametros)
  - [GET /api/](#get-api)
- [Ejemplo de Uso con Postman](#ejemplo-de-uso-con-postman)
- [Notas](#notas)

## Construcción de la Imagen Docker

Para construir la imagen Docker, ejecuta el siguiente comando en el directorio raíz del proyecto (donde se encuentra el `Dockerfile`):

```bash
docker build -t predictive_api .
```

## Ejecución del Contenedor Docker

Después de construir la imagen, puedes ejecutar el contenedor con el siguiente comando:

```bash
docker run -d -p 8887:8887 predictive_api
```

## Endpoints de la API

### POST /api/parametros/

Este endpoint permite recibir datos en formato JSON.

**Cuerpo de la solicitud**:

```json
{
    "NEds": 5,
    "NActDays": 10,
    "pagesWomen": 3,
    "wikiprojWomen": 4
}
```

**Respuesta**:
- **Código de Estado**: `200 OK`
- **Cuerpo de la respuesta**:

```json
{
    "message": "Datos recibidos correctamente",
    "data": {
        "NEds": 5,
        "NActDays": 10,
        "pagesWomen": 3,
        "wikiprojWomen": 4
    }
}
```

### GET /api/

Este endpoint verifica que la API esté corriendo.

**Respuesta**:
- **Código de Estado**: `200 OK`
- **Cuerpo de la respuesta**:

```json
{
    "message": "La API está corriendo"
}
```

## Ejemplo de Uso con Postman

1. Abre Postman y crea una nueva solicitud.
2. Selecciona el método **POST**.
3. Ingresa la URL: `http://localhost:8887/api/parametros/`.
4. En la pestaña **Body**, selecciona **raw** y elige **JSON**.
5. Ingresa los datos JSON en el área de texto y haz clic en **Send**.

## Notas

- Asegúrate de que el contenedor de Docker esté en ejecución antes de enviar solicitudes a la API.
- Los logs del contenedor pueden ser revisados con el comando:

```bash
docker logs <nombre_del_contenedor>
```
