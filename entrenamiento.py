from ultralytics import YOLO

def main():
    # Ruta del dataset
    data_yaml = "dataset/data.yaml"  # ajusta según tu carpeta

    # Modelo base
    model = YOLO("yolov8n.pt")  # pequeño, ideal para CPU

    # Entrenamiento inicial
    results = model.train(
        data=data_yaml,
        epochs=40,        # primera fase
        imgsz=416,
        batch=4,
        workers=1,
        device='cpu',
        name="entrenamiento_fase1"
    )

    print("\n✅ Entrenamiento completado (Fase 1).")
    print("📦 Resultados guardados en:", results.save_dir)
    

if __name__ == "__main__":
    main()
