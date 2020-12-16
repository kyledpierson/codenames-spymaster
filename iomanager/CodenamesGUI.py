import numpy as np
import tkinter as tk

from gamecomponents.Card import Card, Team
from globalvariables import GRID_SIZE

Grid = np.array


class CodenamesGUI:
    def __init__(self):
        pass

    @staticmethod
    def getColor(team: Team) -> str:
        if team == Team.ASSASSIN:
            return "gray10"
        elif team == Team.BLUE:
            return "dodger blue"
        elif team == Team.RED:
            return "firebrick1"
        elif team == Team.NEUTRAL:
            return "bisque"

    def verifyKeyCard(self, cardGrid: Grid):
        window = tk.Tk()

        def changeTeam(event: tk.EventType, i: int, j: int):
            newTeam: int = int(cardGrid[i, j].team) + 1
            if newTeam > 3:
                newTeam = 0
            cardGrid[i, j].setTeam(Team(newTeam))
            newBg: str = CodenamesGUI.getColor(Team(newTeam))
            event.widget.configure(bg=newBg)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                bg: str = CodenamesGUI.getColor(cardGrid[row, col].team)
                frame: tk.Frame = tk.Frame(master=window, width=100, height=100)
                button: tk.Button = tk.Button(master=frame, bg=bg)

                frame.grid_propagate(False)
                frame.columnconfigure(0, weight=1)
                frame.rowconfigure(0, weight=1)
                frame.grid(row=row, column=col)

                button.bind("<Button-1>", lambda event, i=row, j=col: changeTeam(event, i, j))
                button.grid(sticky="wens")

        frame: tk.Frame = tk.Frame(master=window)
        frame.grid(row=5, column=2)
        button: tk.Button = tk.Button(master=frame, text="Submit", command=lambda: window.destroy())
        button.pack()

        window.mainloop()

    def verifyWordGrid(self, cardGrid: Grid):
        window = tk.Tk()
        entries: Grid = Grid([[tk.Entry() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=tk.Entry)

        def changeText():
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    cardGrid[row, col].setText(entries[row, col].get())
            window.destroy()

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                frame: tk.Frame = tk.Frame(master=window)
                entries[row, col] = tk.Entry(master=frame)
                entries[row, col].insert(0, cardGrid[row, col].text)
                frame.grid(row=row, column=col)
                entries[row, col].pack()

        frame: tk.Frame = tk.Frame(master=window)
        frame.grid(row=5, column=2)
        button: tk.Button = tk.Button(master=frame, text="Submit", command=lambda: changeText())
        button.pack()

        window.mainloop()
