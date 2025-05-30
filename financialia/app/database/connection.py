from dotenv import load_dotenv
import os
import asyncpg

class Database:
    def __init__(self):
        # Cargar las variables de entorno desde el archivo .env
        load_dotenv()

        self.BD_USER = os.getenv("DB_USER")
        self.BD_PASSWORD = os.getenv("DB_PASSWORD")
        self.DB_HOST = os.getenv("DB_HOST")
        self.DB_PORT = int(os.getenv("DB_PORT"))
        self.DB_NAME = os.getenv("DB_NAME")

    async def get_connection(self):
        try:
            conn = await asyncpg.connect(
                user=self.BD_USER,
                password=self.BD_PASSWORD,
                host=self.DB_HOST,
                port=self.DB_PORT,
                database=self.DB_NAME
            )
            return conn
        except Exception as e:
            print("Error de conexión:", e)
