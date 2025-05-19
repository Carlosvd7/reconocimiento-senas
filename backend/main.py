import os


#MENU PARA EJECUTAR LOS SCRIPTS QUE TENEMOS

def menu():
    while True:

        print("\n Menú Principal")
        print("1. Capturar nuevos datos de gestos")
        print("2. Entrenar modelo")
        print("3. Ejecutar reconocimiento en tiempo real")
        print("4. Salir")
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            os.system("python3 scripts/capture_data.py")
        elif opcion == "2":
            os.system("python3 scripts/train_model.py")
        elif opcion == "3":
            os.system("python3 scripts/predict_live.py")
        elif opcion == "4":
            break
        else:
            print("❌ Opción no válida. Intenta de nuevo.")

menu()
