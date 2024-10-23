from reactpy import component, html, run, use_state
import requests

@component
def input_form():
    data = {
        "NEds": 0.0,
        "NActDays": 0.0,
        "pagesWomen": 0.0,
        "wikiprojWomen": 0.0
    }
    
    response_data, set_response_data = use_state(None)
    error_message, set_error_message = use_state("")  # Estado para manejar el mensaje de error

    async def submit_data(event):
        api_url = "http://localhost:8887/api/parametros/"  # Cambia a tu API externa
        response = requests.post(api_url, json=data)
        result = response.json()
        set_response_data(result)

    def update_value(key, value):
        try:
            # Intenta convertir el valor a float
            data[key] = float(value) if value else 0.0
            set_error_message("")  # Limpia el mensaje de error si la conversión es exitosa
        except ValueError:
            set_error_message(f"El valor ingresado para {key} no es válido. Debe ser un número.")

    return html.div(
        html.h1("Aplicación para Predecir Genero en Wikipedia 🤖🚀"),
        html.form(
            html.label("NEds: "),
            html.input({
                "type": "number", 
                "step": "any",  # Permitir decimales
                "on_change": lambda event: update_value("NEds", event["target"]["value"])
            }),
            html.br(),
            html.label("NActDays: "),
            html.input({
                "type": "number", 
                "step": "any",
                "on_change": lambda event: update_value("NActDays", event["target"]["value"])
            }),
            html.br(),
            html.label("PagesWomen: "),
            html.input({
                "type": "number", 
                "step": "any",
                "on_change": lambda event: update_value("pagesWomen", event["target"]["value"])
            }),
            html.br(),
            html.label("WikiprojWomen: "),
            html.input({
                "type": "number", 
                "step": "any",
                "on_change": lambda event: update_value("wikiprojWomen", event["target"]["value"])
            }),
            html.br(),
            html.button({"type": "button", "on_click": submit_data}, "Enviar"),
        ),
        # Mostrar el mensaje de error si existe
        error_message and html.div(
            html.h2("Error:", style={"color": "red"}),
            html.p(error_message)
        ),
        # Mostrar la respuesta de la API
        response_data and html.div(
            html.h2(response_data.get("message", "Respuesta de la API:")),  # Usar .get() para evitar KeyError
            html.pre(f"Datos: {response_data.get('data', 'No hay datos disponibles o Coloca Valores Enteros')}")
        ) or html.div(" ")
    )

# Ejecutar el componente
run(input_form)
