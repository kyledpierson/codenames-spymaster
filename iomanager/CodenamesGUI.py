import numpy as np
import tkinter as tk

from gamecomponents.Card import Card, Team
from globalvariables import GRID_SIZE

Grid = np.array


class CodenamesGUI:
    def __init__(self, team: Team = Team.RED):
        self.team: Team = team
        self.gameOver: bool = False

    def captureKeycard(self):
        pass

    def verifyKeycard(self, cardGrid: Grid) -> Team:
        window = tk.Tk()
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                card: Card = cardGrid[row, col]
                bg: str = CodenamesGUI.__getColor(card.team)
                button: tk.Button = tk.Button(master=window, width=12, height=4, bg=bg)
                button.bind("<Button-1>", lambda event, c=card: CodenamesGUI.__changeTeam(event, c))
                button.grid(row=row, column=col, sticky="wens")

        # My team button
        bg = CodenamesGUI.__getColor(self.team)
        button = tk.Button(master=window, width=12, height=4, bg=bg, text="My team")
        button.bind("<Button-1>", self.__changeMyTeam)
        button.grid(row=GRID_SIZE, column=1, sticky="wens")

        # Submit button
        button = tk.Button(master=window, width=12, height=4, text="Submit")
        button.bind("<Button-1>", CodenamesGUI.__close)
        button.grid(row=GRID_SIZE, column=GRID_SIZE - 2, sticky="wens")

        window.mainloop()
        return self.team

    def captureWordgrid(self):
        pass

    def verifyWordgrid(self, cardGrid: Grid):
        window = tk.Tk()
        entries: Grid = Grid([[tk.Entry() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=tk.Entry)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                entries[row, col] = tk.Entry(master=window)
                entries[row, col].insert(0, cardGrid[row, col].text)
                entries[row, col].grid(row=row, column=col)
        button: tk.Button = tk.Button(master=window, text="Submit")
        button.bind("<Button 1>", lambda event: CodenamesGUI.__changeText(event, entries, cardGrid))
        button.grid(row=GRID_SIZE, column=0, columnspan=GRID_SIZE, sticky="wens")
        window.mainloop()

    def displayClueAndWait(self, clue: str) -> bool:
        window = tk.Tk()
        self.gameOver = False

        clueLabel: tk.Label = tk.Label(master=window, text=clue, font=("Helvetica", 24))
        clueLabel.grid(row=0, column=0, columnspan=2, sticky="wens")

        continueButton: tk.Button = tk.Button(master=window, text="Start next turn")
        continueButton.grid(row=1, column=0)
        continueButton.bind("<Button 1>", CodenamesGUI.__close)

        endButton: tk.Button = tk.Button(master=window, text="End game")
        endButton.grid(row=1, column=1)
        endButton.bind("<Button 1>", self.__endGame)

        window.mainloop()
        return self.gameOver

    # ========== PRIVATE ========== #
    @staticmethod
    def __getColor(team: Team) -> str:
        if team == Team.ASSASSIN:
            return "gray10"
        elif team == Team.BLUE:
            return "dodger blue"
        elif team == Team.RED:
            return "firebrick1"
        elif team == Team.NEUTRAL:
            return "bisque"

    @staticmethod
    def __changeTeam(event: tk.EventType, card: Card):
        newTeam: int = int(card.team) + 1
        if newTeam > 3:
            newTeam = 0
        card.team = Team(newTeam)
        newBg: str = CodenamesGUI.__getColor(card.team)
        event.widget.configure(bg=newBg)

    def __changeMyTeam(self, event: tk.EventType):
        self.team = Team.RED if self.team is Team.BLUE else Team.BLUE
        newBg: str = CodenamesGUI.__getColor(self.team)
        event.widget.configure(bg=newBg)

    @staticmethod
    def __close(event: tk.EventType):
        event.widget.master.destroy()

    @staticmethod
    def __changeText(event: tk.EventType, entries: Grid, cardGrid: Grid):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                text: str = entries[row, col].get().strip().replace(" ", "_")
                cardGrid[row, col].text = text
                cardGrid[row, col].visible = bool(text)
        CodenamesGUI.__close(event)

    def __endGame(self, event: tk.EventType):
        self.gameOver = True
        CodenamesGUI.__close(event)
