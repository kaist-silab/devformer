import streamlit as st
from PIL import Image


st.write("# Problem Introduction")


st.write(
    r"""
The performance of DPP is evaluated by power integrity (PI) simulation that computes the level of impedance suppression over a specified frequency domain and is quantified as:
"""
)

st.latex(
    r"""
    \mathcal{J} :=\sum_{f \in F} (Z_{initial}(f)-Z_{final}(f)) \cdot \frac{\text{1GHz}}{f}
"""
)

st.write(
    r"""
where $Z_{initial}$ and $Z_{final}$ are the initial and final impedance at the frequency $f$ before and after placing decaps, respectively. $F$ is the set of specified frequency points. The PI simulation for $(N_{row} \times N_{col})$ PDN requires a $N_{row}N_{col} \times N_{row}N_{col} \times n_{freq}$ (number of frequency points) sized Z-parameter matrix calculation because each port is electrically coupled to the rest of the ports and Z (i.e., impedance) is frequency-dependent. Thus, performance evaluation with a large Z-parameter matrix calculation is costly. The more impedance is suppressed, the better the power integrity and the higher the performance score. Note that this performance metric was also used for collecting the offline expert data using a genetic algorithm (GA). 
"""
)

image = Image.open("pages/assets/figure3.jpg")


st.image(image, caption="Unit-cell of real-world target PDN")
