import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os
import ctypes
from tkinter import ttk  # Para la tabla


# Ruta al modelo
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "blackbox_S.keras")
model = load_model(model_path)

def predict_point(model, x1, x2):
    data = np.array([[x1, x2]])
    prediction = model.predict(data, verbose=0)[0][0]
    return int(prediction > 0.5), prediction

def predict_batch(model, x1_list, x2_list):
    data = np.column_stack((x1_list, x2_list))
    predictions = model.predict(data, verbose=0)
    return (predictions > 0.5).astype(int).flatten().tolist()

# Ventana principal con menú
def mostrar_menu():
    
    menu = tk.Tk()

    # Centrar la ventana en la pantalla
    menu.update_idletasks()
    menu.title("Menú - Black Box")
    ancho_ventana = 600
    alto_ventana = 500
    ancho_pantalla = menu.winfo_screenwidth()
    alto_pantalla = menu.winfo_screenheight()
    x = (ancho_pantalla // 2) - (ancho_ventana // 2)
    y = (alto_pantalla // 2) - (alto_ventana // 2)
    menu.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

    # Fondo con imagen
    try:
    # Usar imagen de fondo desde la carpeta assets
        img_path = os.path.join(script_dir, "assets", "fondo2.png")
        img = Image.open(img_path)
        fondo = ImageTk.PhotoImage(img.resize((600, 500)))
        label_fondo = tk.Label(menu, image=fondo)
        label_fondo.place(x=0, y=0, relwidth=1, relheight=1)
        label_fondo.image = fondo
    except:
        menu.configure(bg="#2F2F2F")

    def abrir_aplicacion():
        menu.destroy()
        lanzar_clasificador()

    def mostrar_mensaje(titulo, texto):
        messagebox.showinfo(titulo, texto)
    
    # Título del menú
    tk.Label(menu, text="Menú Principal - Black Box", font=("Helvetica", 16, "bold"), bg="#DDEEFF", fg="#333").pack(pady=20)


    # Botones del menú principal
    tk.Button(menu, text="Iniciar Aplicación", font=("Helvetica", 12, "bold"), bg="#424242", fg="white", command=abrir_aplicacion, width=30).pack(pady=12)

    tk.Button(menu, text="¿Qué es Black Box?", font=("Helvetica", 12), command=lambda: mostrar_mensaje("¿Qué es Black Box?", "Black Box es un modelo entrenado que clasifica puntos en 2D (x₁, x₂) en clases binarias usando una red neuronal."), width=30).pack(pady=8)

    tk.Button(menu, text="Información de la Aplicación", font=("Helvetica", 12), command=lambda: mostrar_mensaje("Información", "Aplicación interactiva que predice visualmente la clase de un punto en un plano bidimensional y muestra la frontera de decisión aprendida."), width=30).pack(pady=8)

    tk.Button(menu, text="Créditos", font=("Helvetica", 12), command=lambda: mostrar_mensaje("Créditos", "Desarrollado por: Dany Molina, Joseph Jimenez, Kevin Gomez \nEscuela Poliecnica Nacional - 2025"), width=30).pack(pady=8)
   
    tk.Button(menu, text="Salir", command=menu.quit, font=("Helvetica", 11), bg="red", fg="white", width=15).pack(pady=10)

    menu.mainloop()
# Lista global para almacenar el historial
historial = []

# VENTANA SECUNDARIA
def mostrar_tabla_resultados():
    ventana_tabla = tk.Toplevel()
    ventana_tabla.title("Tabla de Resultados")
    ventana_tabla.geometry("500x400")
    ventana_tabla.configure(bg="#2F2F2F")

    # Centrar la ventana
    ventana_tabla.update_idletasks()
    ancho_ventana = 500
    alto_ventana = 400
    ancho_pantalla = ventana_tabla.winfo_screenwidth()
    alto_pantalla = ventana_tabla.winfo_screenheight()
    x = (ancho_pantalla // 2) - (ancho_ventana // 2)
    y = (alto_pantalla // 2) - (alto_ventana // 2)
    ventana_tabla.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

    tk.Label(ventana_tabla, text="Historial de Predicciones", font=("Helvetica", 14, "bold"), bg="#F2F2F2").pack(pady=10)

    tabla = ttk.Treeview(ventana_tabla, columns=("x1", "x2", "clase", "prob"), show="headings")
    tabla.heading("x1", text="x₁")
    tabla.heading("x2", text="x₂")
    tabla.heading("clase", text="Clase")
    tabla.heading("prob", text="Probabilidad")

    # --- Validar datos únicos antes de insertar ---
    datos_unicos = set()  # Conjunto para evitar duplicados

    for dato in historial:
        if tuple(dato) not in datos_unicos:
            datos_unicos.add(tuple(dato))
            tabla.insert('', 'end', values=dato)
        # Si el dato ya existe, no lo inserta

    tabla.pack(pady=10)

    explicacion = (
        "Esta tabla almacena los valores ingresados (x₁, x₂), la clase predicha por el modelo\n"
        "y la probabilidad obtenida. El modelo es una red neuronal que clasifica los puntos\n"
        "en 2D en dos clases (0 o 1), dependiendo de su ubicación relativa en el plano."
    )

    tk.Label(ventana_tabla, text=explicacion, bg="#F2F2F2", fg="#444", font=("Helvetica", 10), justify="left").pack(padx=10, pady=10)

# VENTANA PRINCIPAL
def lanzar_clasificador():
    root = tk.Tk()
    root.title("Clasificador con Modelo Keras")
    root.configure(bg="#FFFFFF")
    # Centrar la ventana en la pantalla
    root.update_idletasks()
    ancho_ventana = 1000
    alto_ventana = 600
    ancho_pantalla = root.winfo_screenwidth()
    alto_pantalla = root.winfo_screenheight()
    x = (ancho_pantalla // 2) - (ancho_ventana // 2)
    y = (alto_pantalla // 2) - (alto_ventana // 2)
    root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")
    # FRAME IZQUIERDO (SECCIÓN 2) - Entrada de datos y botones
    frame_izquierda = tk.Frame(root, bg="#404040", padx=10, pady=10)
    # Imagen de fondo para el frame izquierdo
    try:
        bg_img_path = os.path.join(script_dir, "assets", "left_bg.png")
        bg_img = Image.open(bg_img_path)
        bg_img = bg_img.resize((300, 700))
        frame_izquierda.config(width=300, height=700)
        frame_izquierda.pack_propagate(False)
        bg_photo = ImageTk.PhotoImage(bg_img)
        bg_label = tk.Label(frame_izquierda, image=bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.image = bg_photo
    except Exception:
        frame_izquierda.configure(bg="#404040")
    frame_izquierda.pack(side="left", fill="y")

    tk.Label(frame_izquierda, text="Ingrese x₁:", font=("Helvetica", 12), bg="#F2F2F2").pack(pady=(5, 0))
    entry_x1 = tk.Entry(frame_izquierda, font=("Helvetica", 12))
    entry_x1.pack()

    tk.Label(frame_izquierda, text="Ingrese x₂:", font=("Helvetica", 12), bg="#F2F2F2").pack(pady=(10, 0))
    entry_x2 = tk.Entry(frame_izquierda, font=("Helvetica", 12))
    entry_x2.pack()

    label_resultado = tk.Label(frame_izquierda, text="Predicción del modelo:", font=("Helvetica", 12, "bold"), bg="#F2F2F2")
    label_resultado.pack(pady=10)

    def predecir():
        try:
            # Leer lo ingresado en los campos
            x1_input = entry_x1.get()
            x2_input = entry_x2.get()

            # Si hay comas, interpretamos como múltiples valores
            if ',' in x1_input or ',' in x2_input:
                # Convertir los valores a listas de flotantes
                x1_list = [float(x.strip()) for x in x1_input.split(',')]
                x2_list = [float(x.strip()) for x in x2_input.split(',')]

                # Verificar que tengan la misma longitud
                if len(x1_list) != len(x2_list):
                    messagebox.showerror("Error", "Las listas x₁ y x₂ deben tener la misma longitud.")
                    return

                # Usar predict_point para cada par
                resultados = [predict_point(model, x1, x2) for x1, x2 in zip(x1_list, x2_list)]

                # Mostrar el último resultado en la etiqueta
                clase, probabilidad = resultados[-1]
                label_resultado.config(text=f"Última predicción: Clase {clase} (Prob: {probabilidad:.2f})")

                # Guardar en historial y graficar todos los puntos
                historial.extend([
                    (f"{x1:.2f}", f"{x2:.2f}", str(cl), f"{prob:.2f}")
                    for (x1, x2), (cl, prob) in zip(zip(x1_list, x2_list), resultados)
                ])

            else:
                # Si no hay comas, tratamos como un solo punto
                x1 = float(x1_input)
                x2 = float(x2_input)
                clase, probabilidad = predict_point(model, x1, x2)
                label_resultado.config(text=f"Predicción: Clase {clase} (Prob: {probabilidad:.2f})")
                historial.append((f"{x1:.2f}", f"{x2:.2f}", str(clase), f"{probabilidad:.2f}"))
                x1_list = [x1]
                x2_list = [x2]

            # Generar gráfica
            # Determinar rangos dinámicos
            x1_min = min(0.0, min(x1_list))  # Siempre x1 ≥ 0, pero en caso de que el punto sea > 2.0
            x1_max = max(2.0, max(x1_list) + 0.2)  # Agregar un margen
            x2_min = min(-0.5, min(x2_list) - 0.2)
            x2_max = max(2.0, max(x2_list) + 0.2)

            # Crear malla de puntos
            x1_vals = np.linspace(x1_min, x1_max, 100)
            x2_vals = np.linspace(x2_min, x2_max, 100)
            X1, X2 = np.meshgrid(x1_vals, x2_vals)

            # Obtener predicciones en la grilla
            Z = predict_batch(model, X1.ravel(), X2.ravel())
            Z = np.array(Z).reshape(X1.shape)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.contourf(X1, X2, Z, levels=[-0.1, 0.5, 1.1], cmap="coolwarm", alpha=0.8)
            ax.plot(x1_list, x2_list, 'ko', label='Punto(s) ingresado(s)')
            ax.set_xlim(x1_min, x1_max)
            ax.set_ylim(x2_min, x2_max)
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_title("Mapa de predicción del modelo")
            ax.grid(True)
            ax.legend()

            # Mostrar en la GUI
            for widget in frame_derecha.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=frame_derecha)
            canvas.draw()
            canvas.get_tk_widget().pack()
            plt.close(fig)


        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa valores numéricos válidos (usa ',' para múltiples puntos).")
    def detectar_frontera(x_fixed, y_range, is_vertical):
        frontera_x, frontera_y = [], []
        anterior = None
        for y in y_range:
            x = x_fixed if is_vertical else y
            y_val = y if is_vertical else x_fixed
            clase = predict_point(model, x, y_val)[0]  # Extraer solo la clase
            if anterior is not None and clase != anterior:
                frontera_x.append(x)
                frontera_y.append(y_val)
            anterior = clase
        return frontera_x, frontera_y

    def graficar_lote():
        try:
            x1_input = entry_x1.get()
            x2_input = entry_x2.get()

            if not x1_input or not x2_input:
                messagebox.showerror("Error", "Por favor ingrese valores en los campos x₁ y x₂.")
                return

            lote_x1 = [float(x.strip()) for x in x1_input.split(',')]
            lote_x2 = [float(x.strip()) for x in x2_input.split(',')]

            if len(lote_x1) != len(lote_x2):
                messagebox.showerror("Error", "Las listas x₁ y x₂ deben tener la misma longitud.")
                return

            clases = predict_batch(model, lote_x1, lote_x2)

            # --- Rango dinámico
            x1_min, x1_max = min(0.0, min(lote_x1)), max(2.0, max(lote_x1) + 0.5)
            x2_min, x2_max = min(-0.5, min(lote_x2) - 0.5), max(2.0, max(lote_x2) + 0.5)

            # --- Malla reducida para acelerar (100x100)
            x1_vals = np.linspace(x1_min, x1_max, 100)
            x2_vals = np.linspace(x2_min, x2_max, 100)
            X1, X2 = np.meshgrid(x1_vals, x2_vals)
            Z = np.array(predict_batch(model, X1.ravel(), X2.ravel())).reshape(X1.shape)

            # --- Frontera optimizada
            step = 0.05  # Mayor valor = menos puntos frontera = más rápido
            fx1, fy1 = detectar_frontera(x1_min, np.arange(x2_min, x2_max, step), is_vertical=True)
            fx2, fy2 = detectar_frontera(x1_max, np.arange(x2_min, x2_max, step), is_vertical=True)
            fx3, fy3 = detectar_frontera(x2_min, np.arange(x1_min, x1_max, step), is_vertical=False)
            fx4, fy4 = detectar_frontera(x2_max, np.arange(x1_min, x1_max, step), is_vertical=False)

            frontera_x = fx1 + fx2 + fx3 + fx4
            frontera_y = fy1 + fy2 + fy3 + fy4

            # --- Crear gráfico
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.contourf(X1, X2, Z, levels=[-0.1, 0.5, 1.1], cmap="coolwarm", alpha=0.7)
            ax.set_xlim(x1_min, x1_max)
            ax.set_ylim(x2_min, x2_max)
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_title("Mapa de predicción con puntos ingresados y frontera")
            ax.grid(True)

            # Agrupar y dibujar puntos por clase (para evitar múltiples `scatter`)
            puntos_clase1 = [(x, y) for x, y, c in zip(lote_x1, lote_x2, clases) if c == 1]
            puntos_clase0 = [(x, y) for x, y, c in zip(lote_x1, lote_x2, clases) if c == 0]

            if puntos_clase1:
                x1, y1 = zip(*puntos_clase1)
                ax.scatter(x1, y1, c='black', edgecolors='white', s=80, label='Clase 1')
            if puntos_clase0:
                x0, y0 = zip(*puntos_clase0)
                ax.scatter(x0, y0, c='white', edgecolors='black', s=80, label='Clase 0')

            # Puntos de frontera
            if frontera_x and frontera_y:
                ax.scatter(frontera_x, frontera_y, c='yellow', edgecolors='black', s=40, alpha=0.8, label='Frontera')

            # Eliminar duplicados en leyenda
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())

            # Actualizar el frame derecho
            for widget in frame_derecha.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=frame_derecha)
            canvas.draw()
            canvas.get_tk_widget().pack()
            plt.close(fig)

        except ValueError:
            messagebox.showerror("Error", "Asegúrate de ingresar solo números separados por comas.")

    def limpiar():
        entry_x1.delete(0, tk.END)
        entry_x2.delete(0, tk.END)
        label_resultado.config(text="Predicción del modelo:")

    def volver_menu():
        root.destroy()
        mostrar_menu()

    tk.Button(frame_izquierda, text="Predecir y Graficar", command=predecir, bg="#007ACC", fg="white", font=("Helvetica", 12)).pack(pady=10)
    tk.Button(frame_izquierda, text="Limpiar Entradas", command=limpiar, bg="orange", font=("Helvetica", 11)).pack(pady=5)
    tk.Button(frame_izquierda, text="Graficar Lote de Puntos", command=graficar_lote, bg="#4CAF50", fg="white", font=("Helvetica", 12)).pack(pady=10)
    tk.Button(frame_izquierda, text="Ver Tabla de Resultados", command=mostrar_tabla_resultados, bg="green", fg="white", font=("Helvetica", 11)).pack(pady=5)
    tk.Button(frame_izquierda, text="Volver al Menú", command=volver_menu, bg="gray", fg="white", font=("Helvetica", 11)).pack(pady=5)

    # FRAME DERECHO (SECCIÓN 1) - Gráfica
    frame_derecha = tk.Frame(root, bg="#2A2A2A", padx=10, pady=10)
    # Imagen de fondo para el frame derecho
    try:
        right_bg_img_path = os.path.join(script_dir, "assets", "right_bg.png")
        right_bg_img = Image.open(right_bg_img_path)
        right_bg_img = right_bg_img.resize((700, 600))
        right_bg_photo = ImageTk.PhotoImage(right_bg_img)
        right_bg_label = tk.Label(frame_derecha, image=right_bg_photo)
        right_bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        right_bg_label.image = right_bg_photo
    except Exception:
        frame_derecha.configure(bg="#2A2A2A")
    frame_derecha.pack(side="right", fill="both", expand=True)

    root.mainloop()
# Iniciar menú principal
mostrar_menu()
