# Erkennung von Amerikanischer Zeichensprache (ASL) mit Keras
## Projektbeschreibung

Dieses Projekt ist ein Beispiel für die Erkennung von Amerikanischer Zeichensprache (ASL) mit Keras. Es wurde im Rahmen des Moduls "Introduction to Data Science" an der DHBW Lörrach erstellt.

Das Projekt besteht aus folgenden Komponenten:
* Jupyter Notebook mit dem Code
* Trainingsdaten
* Testdaten
* Modelldatei
* README.md
* requirements.txt 
* Output Directory

## Installation

Das Projekt läuft mit Python 3.9. Es wird empfohlen, ein virtuelles Environment zu verwenden. Die Erstellung des virtuellen Environment mit Conda und die Installation der benötigten Pakete lässt sich über die folgenden Befehle bewerkstelligen.

```bash
  conda create -n  ml_intro_project python=3.9
```

```bash
conda activate ml_intro_project
conda install --file requirements.txt
pip install jupyter
```

## Ausführung
Das Model ist bereits trainiert und kann direkt verwendet werden. Dazu muss lediglich das Jupyter Notebook ausgeführt werden. Das Model kann neu trainiert werden, indem die Variable Retrain im Jupyter Notebook zu True geändert wird. Weitere Dokumentation zum Code lässt sich im Jupyter Notebook finden.