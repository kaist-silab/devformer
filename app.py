import streamlit as st
import streamlit.components.v1 as components
import webbrowser

from src.problems.dpp.simulator import decap_sim
from download_anonymous_github import download_repo


APP_NAME = "DPP Benchmark"

N_COLS = 10
N_ROWS = 10
PADDING_X = 1
PADDING_Y = 2

# Make dictionary of states with respective colors
states_dict = {
    "AVAILABLE": {"text": "⬜", "color": "white"},
    "SOLUTION": {"text": "🔵", "color": "blue"},
    "PROBE": {"text": "🟥", "color": "red"},
    "KEEPOUT": {"text": "✖️", "color": "black"},
}

# NOTE: replaced with grey background for now, way faster
IMAGE = "https://media.istockphoto.com/id/899824734/vector/digital-circuit-background-texture-of-processor-motherboard.jpg?s=612x612&w=0&k=20&c=25oCxPTkika0jny7LvfOWGCGsDjNV8zKcruHk-Mf9rU="
BACKGROUND_COLOR = "#808080"
OPACITY = 0.8
BUTTON_SCALE = 1.2
MIN_WIDTH = 800


st.set_page_config(page_title=APP_NAME, page_icon=":zap:")
st.title(f"Decoupling Capacitor Placement")

# Add permanent notice
st.markdown(
    """
    <div style="background-color:#f8f8f8; border-radius: 5px;">
    <p>
    <b>Note</b>: we recommend using the following website in desktop mode - it is not optimized for mobile devices; the layout might be broken in the mobile version.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Title
st.markdown(
    f"""
    To download from Anonymous Github automatically, click on "Downloader Script" and run:
    `python3 download_anonymous_github.py`
    in the terminal. This will download the repository to the current directory.
    """
)

col1, col2 = st.columns([1, 1])


# Download button for anonymous github
with open("download_anonymous_github.py", "r") as f:
    download_script = f.read()

col1.download_button(
    label="Downloader Script ⬇️",
    data=download_script,
    file_name="download_anonymous_github.py",
)

if col2.button('Code from Anonymous Github ↗️', key="download", type="secondary"):
    webbrowser.open_new_tab("https://anonymous.4open.science/r/DPPBench")

# Divider
st.markdown("---")

# Initialization
if "buttons" not in st.session_state:
    st.session_state["buttons"] = [["AVAILABLE"] * N_COLS for i in range(N_ROWS)]
if "mode" not in st.session_state:
    st.session_state["mode"] = "PROBE"
if "score" not in st.session_state:
    st.session_state["score"] = 0
if "maximum_score" not in st.session_state:
    st.session_state["maximum_score"] = 0
if "num_decap" not in st.session_state:
    st.session_state["num_decap"] = 0
if "maximum_score_decap" not in st.session_state:
    st.session_state["maximum_score_decap"] = 0


############################
## Functions
############################

# make any grid with a function
def make_grid(cols, rows, padding_x=PADDING_X, padding_y=PADDING_Y):
    cols_ = cols + 2 * padding_x
    rows_ = rows + 2 * padding_y
    grid = [0] * rows_
    for i in range(rows_):
        with st.container():
            grid[i] = st.columns(cols_)
    return grid


def reset_grid():
    st.session_state["buttons"] = [["AVAILABLE"] * N_ROWS for i in range(N_COLS)]


def change_mode(mode):
    st.session_state["mode"] = mode


def check_results(n_cols=N_COLS, n_rows=N_ROWS):
    """Check results and update score"""
    # Gt probe, solution, keepout from session state
    probe = []
    solution = []
    keepout = []
    for i in range(n_cols):
        for j in range(n_rows):
            idx = i * n_cols + j
            if st.session_state["buttons"][i][j] == "PROBE":
                probe.append(idx)
            elif st.session_state["buttons"][i][j] == "SOLUTION":
                solution.append(idx)
            elif st.session_state["buttons"][i][j] == "KEEPOUT":
                keepout.append(idx)

    # Checks
    if len(probe) > 1:
        st.warning("Only one probe is allowed!", icon="⚠️")
        return
    # Check if we have at least one probe and one solution
    if len(probe) == 0:
        st.warning("No probe is placed!", icon="⚠️")
        return
    if len(solution) == 0:
        st.warning("No solution is placed!", icon="⚠️")
        return

    # Check solution cost
    score = decap_sim(probe=probe[0], solution=solution, keepout=keepout)
    st.session_state["score"] = score
    st.session_state["num_decap"] = len(solution)

    # Mew best
    if score > st.session_state["maximum_score"]:
        st.session_state["maximum_score"] = score
        st.session_state["maximum_score_decap"] = st.session_state["num_decap"]
        st.success("New best score: {:.3f}".format(float(score)), icon="🎉")
        st.balloons()
    else:
        st.info("Current score: {:.3f}".format(float(score)))


def button_clicked(i, j):
    """Action when button is clicked"""
    # If button is not available, make available
    if st.session_state["buttons"][i][j] != "AVAILABLE":
        st.session_state["buttons"][i][j] = "AVAILABLE"
        return
    # Count probes, if more than one, warning and return
    if st.session_state["mode"] == "PROBE":
        count = 0
        for _i in range(N_COLS):
            for _j in range(N_ROWS):
                if st.session_state["buttons"][_i][_j] == "PROBE":
                    count += 1
        if count >= 1:
            st.warning("Only one probe is allowed!", icon="⚠️")
            return
    st.session_state["buttons"][i][j] = st.session_state["mode"]


############################
## Main Python Layout
############################
#
# # st.text(f"Placing a {st.session_state['mode'].lower()}...")

st.write(
    f"**Task**: given a configuration of probe {states_dict['PROBE']['text']} and keepout {states_dict['KEEPOUT']['text']}, place the decoupling capacitor {states_dict['SOLUTION']['text']} such that the total cost is minimized."
)

cols = st.columns(3)
with cols[0]:
    st.button(
        f"Place Probe {states_dict['PROBE']['text']}",
        on_click=change_mode,
        args=("PROBE",),
        type="secondary",
    )
with cols[1]:
    st.button(
        f"Place Decap {states_dict['SOLUTION']['text']}",
        on_click=change_mode,
        args=("SOLUTION",),
        type="secondary",
    )
with cols[2]:
    st.button(
        f"Place Keepout {states_dict['KEEPOUT']['text']}",
        on_click=change_mode,
        args=("KEEPOUT",),
        type="secondary",
    )

st.info(
    f"Placing a {st.session_state['mode'].lower()} {states_dict[st.session_state['mode']]['text']}..."
)

st.markdown("----")
with st.container():
    grid = make_grid(N_COLS, N_ROWS)
st.markdown("----")


# Put a button in each grid
for i in range(N_COLS):
    for j in range(N_ROWS):
        idx = i * (N_COLS + PADDING_X * 2) + j + PADDING_X  # padding
        state = st.session_state["buttons"][i][j]
        grid[i + PADDING_Y][j + PADDING_X].button(
            states_dict[state]["text"],
            key=idx,
            on_click=button_clicked,
            args=(i, j),
            help=f"Node {idx} | Location ({i},{j}) - Current state: {state}",
        )

# Put labels on first row and first column
for i in range(N_COLS):
    grid[0][i + PADDING_X].markdown(
        f"<div style='text-align: center'><code>{i}<code></div>", unsafe_allow_html=True
    )

for i in range(N_ROWS):
    grid[i + PADDING_Y][0].markdown(
        f"<div style='text-align: right'><code>{i}<code></div>", unsafe_allow_html=True
    )

cols = st.columns(2)

# Centering the button
with cols[0]:
    st.button(
        "Check score 🔎",
        on_click=check_results,
        help="Check the score of the current configuration",
        type="primary",
    )
with cols[1]:
    st.button("Reset 🔄", on_click=reset_grid, help="Reset the grid", type="primary")

st.info(
    "Score: {:.3f} | Number of decaps: {}".format(
        float(st.session_state["score"]), st.session_state["num_decap"]
    )
)
st.info(
    "Best score: {:.3f} | Number of decaps: {}".format(
        float(st.session_state["maximum_score"]),
        st.session_state["maximum_score_decap"],
    )
)


############################
## Javascript Tricks
############################

replace_dict = {
    "AVAILABLE_TEXT": states_dict["AVAILABLE"]["text"],
    "AVAILABLE_COLOR": states_dict["AVAILABLE"]["color"],
    "PROBE_TEXT": states_dict["PROBE"]["text"],
    "PROBE_COLOR": states_dict["PROBE"]["color"],
    "SOLUTION_TEXT": states_dict["SOLUTION"]["text"],
    "SOLUTION_COLOR": states_dict["SOLUTION"]["color"],
    "KEEPOUT_TEXT": states_dict["KEEPOUT"]["text"],
    "KEEPOUT_COLOR": states_dict["KEEPOUT"]["color"],
    "ELEM_NUM": str(N_COLS * N_ROWS - 1),
    "URL": IMAGE,
    "BUTTON_SCALE": str(BUTTON_SCALE),
    "BACKGROUND_COLOR": BACKGROUND_COLOR,
    "OPACITY": str(OPACITY),
    "MIN_WIDTH": str(MIN_WIDTH),
}

# Load Javascript
with open("pages/src/scripts.js") as f:
    my_js = f.read()
my_js = f"<script>{my_js}</script>"

# Find all elements in the my_js string encapsulated by ${}
for key, value in replace_dict.items():
    my_js = my_js.replace("$" + key, value)

# Inject Javascript
components.html(my_js)
