# Usamos una imagen base ligera de Python 3.11
FROM python:3.11-slim

# Establecemos las variables de entorno para evitar la interacción durante la instalación
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Configuramos la zona horaria a América/Mexico_City
RUN ln -snf /usr/share/zoneinfo/America/Mexico_City /etc/localtime && echo 'America/Mexico_City' > /etc/timezone

# Copiamos el archivo de requisitos e instalamos las dependencias de Python
COPY requirements/requirements.txt /prj/requirements.txt
WORKDIR /prj
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiamos todo el código fuente
COPY . /prj

# Exponemos el puerto 8887
EXPOSE 8887

# Comando por defecto para ejecutar la app
CMD ["python3", "server.py"]
