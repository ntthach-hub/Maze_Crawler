# 🏆 Maze Crawler – Kaggle Competition Agent

AI agent developed for the **Kaggle Maze Crawler competition**. The bot explores an infinite scrolling maze, collects energy efficiently, avoids traps, and survives longer than the opponent under fog-of-war conditions.

<p align="center">
  <img src="Screenshot%202026-07-02%20095301.png" width="700"/>
</p>

---

## 📊 Competition result

<p align="center">
  <img src="images/leaderboard.png" width="900"/>
</p>

| Metric           |    Result |
| ---------------- | --------: |
| **Global Rank**  |  **#206** |
| **Public Score** | **734.5** |
| **Submissions**  |     **2** |

> Achieved **Top 206** on the Kaggle public leaderboard.

---

## 🎯 Strategy overview

The agent focuses on:

* **Exploration** with scouts to reveal new maze areas
* **Energy collection** from crystals
* **Path planning** using remembered wall information
* **Safe movement** to avoid dead ends and enemy collisions
* **Factory survival** while the map continuously scrolls northward

The maze layout is remembered even after leaving vision range, allowing the bot to reuse discovered routes and reduce unnecessary exploration.

---

## 📁 Project structure

```text
Maze_Crawler/
├── main.py              # Main competition agent
├── test_local.py        # Local testing script
├── Screenshot 2026-07-02 095301.png
└── README.md
```

---

## 🚀 Run locally

Install Kaggle Environments:

```bash
pip install kaggle-environments
```

Run a local match:

```bash
python test_local.py
```

---

## 🧠 Key ideas implemented

* Persistent maze memory
* Frontier-based exploration
* Energy-aware movement
* Factory protection logic
* Basic enemy avoidance heuristics

---

## 🔗 Competition

* Kaggle Environment: **Crawl / Maze Crawler**
* This repository contains **only the agent implementation**, not the game engine.

---

## 📌 Notes

The leaderboard result was achieved with the code in `main.py`. Further improvements may include stronger enemy prediction, coordinated multi-scout exploration, and more advanced pathfinding heuristics.
