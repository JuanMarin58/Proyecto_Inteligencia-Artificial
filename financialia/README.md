
# Financialia

**Financialia** es un chatbot financiero desarrollado en Python que responde preguntas sobre inversión en el índice bursátil S&P 500 y cinco de las empresas con mayor capitalización: Google, Microsoft, Apple, Tesla, Amazon e Nvidia.

Además, el chatbot está conectado a un modelo LSTM para la predicción de retornos de estos activos. La aplicación utiliza una base de datos PostgreSQL para almacenamiento, y requiere una configuración local a través de un archivo `.env`.

---

## Características principales

- Chatbot basado en procesamiento de lenguaje natural (NLP) para interacción financiera.
- Modelo LSTM para predicción de retornos bursátiles.
- Consulta información sobre S&P 500 y cinco empresas líderes.
- API construida con FastAPI.
- Interfaz frontend en JavaScript y HTML.
- Base de datos PostgreSQL para manejo de datos.

---

## Tecnologías

- Python  
- FastAPI  
- PostgreSQL  
- JavaScript y HTML  
- Librerías especificadas en `requirements.txt`

---

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/JuanMarin58/Proyecto_Inteligencia-Artificial.git
   cd Proyecto_Inteligencia-Artificial
   ```

2. Crea y activa un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Configura tu base de datos PostgreSQL y crea un archivo `.env` en la raíz del proyecto con las variables necesarias, por ejemplo:

   ```
   POSTGRES_USER=tu_usuario
   POSTGRES_PASSWORD=tu_contraseña
   POSTGRES_DB=tu_basedatos
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   ```

---

## Uso

Para ejecutar la aplicación, corre el servidor FastAPI con Uvicorn desde la carpeta `app/api`:

```bash
cd app/api
python api.py
```

Luego abre tu navegador en [http://localhost:8000](http://localhost:8000) para interactuar con el chatbot y acceder a la API.

---

## Estructura del proyecto

- `app/` — Backend principal en Python que incluye:
  - `api.py`: Punto de entrada de la API con FastAPI.
  - `database/`: Módulo para conexión y manejo de la base de datos PostgreSQL.
  - `model/`: Implementación del modelo LSTM para predicción.
  - `services/`: Lógica de negocio y servicios relacionados al chatbot y predicciones.
  - `utils/`: Funciones utilitarias y helpers.

- `chatbot/` — Módulo de procesamiento de lenguaje natural (NLP) para el chatbot:
  - `chatbot.py`: Código principal del chatbot.
  - Archivos de soporte (como el Excel con preguntas frecuentes).

- `interfaz/` — Frontend de la aplicación:
  - `static/`: Archivos estáticos como CSS, JS, imágenes.
  - `templates/`: Archivos HTML para la interfaz.

- `.env` — Archivo de configuración local con variables de entorno (no incluido en el repositorio).

- `requirements.txt` — Listado de dependencias Python necesarias para el proyecto.

---

## Contribuciones

Las contribuciones son bienvenidas. Por favor abre un issue o un pull request para sugerir mejoras o corregir errores.

---

## Licencia

Este proyecto está bajo licencia MIT. Consulta el archivo `LICENSE` para más detalles.

