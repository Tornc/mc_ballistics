# Create: Big Cannons [Cannon Doxxer](## "*will fail... occasionally")

<blockquote>
  <details>
    <summary><strong>🚧 Status: Work in Progress</strong></summary>
    <strong>Heads up!</strong> The reverse calculator is a bit unreliable. And <code>simulation.py()</code> is an absolute mess.
  </details>
</blockquote> <br>

A web-based ballistics simulation tool for the Create: Big Cannons addon. Visualise trajectories, calculate firing solutions and locate (enemy) cannon locations.

⬇️ Check it out here!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mc-ballistics.streamlit.app/)

---

![pic1](./docs/calculator.png)

<p align="center">
    <em>
        Cannon stats and manual firing option (left) and interactive plot (right).
    </em>
</p>

![pic2](./docs/reverse.png)

<p align="center">
    <em>
        Enemy cannon muzzle locator (left) and firing solution information (right).
    </em>
</p>

## Why?

> Plotly figures are pretty and WebApps are cool.

I guess I got sidetracked 😅.

Anyways, the initial goal was to make a proof of concept that demonstrated the possibility of automated counter-battery systems. This is done through a Some Peripherals' Radar or anything equivalent. The key requirement is that the peripheral has to: **detect entities and record their position**.

## How to use

1. Click on the 'Open in Streamlit' badge.
2. Use the sidebar to set simulation parameters. The 'Calculator' section contains run-of-the-mill features for firing at coordinates. The 'Reverse calculator' is a bit more unique.
3. The main view ('Plot' tab) displays an interactive 3D plot, in which you can move around with the camera.
4. The 'Results' tab shows information required for firing solutions.

## Installation

1. Download the repository.

```
$ git clone https://github.com/Tornc/mc_ballistics.git
```

2. Install the requirements.

```
$ pip install -r requirements.txt
```

3. Run the app.

```cmd
streamlit run app.py
```

Or run `script.cmd` (it does the exact same thing).

4. The app will open in your browser.

## Credits

The trajectory simulation's formulas used have been written by **@sashafiesta**, in the Create: Big Cannons discord. The pitch calculator is based on **@endal**'s Desmos ballistic calculator. I've also been inspired by **@malexy**'s brute-force pitch calculator.

## TODO

- [ ] Clean up the simulation back-end.
- [ ] Option to increase trajectory simulation fidelity (<1 ticks).
- [ ] Get [Stlite](https://github.com/whitphx/stlite/blob/main/packages/desktop/README.md) to work so binaries can be shared. (I still can't get rid of that fugly Electron toolbar.)
