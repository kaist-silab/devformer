const elements = window.parent.document.querySelectorAll('[data-testid="stVerticalBlock"]');
    
let found = [];
for (let i = 0; i < elements.length; i++) {
    const element = elements[i];
    const buttons = element.querySelectorAll("button");
    if (buttons.length > $ELEM_NUM ) {
        found.push(element);
    }
}

container = found[found.length-1];
container.style.backgroundImage =  "url($URL)";
container.style.opacity = "$OPACITY";


function changeButtonStyle(button) {
    button.style.transform = "scale($BUTTON_SCALE)";
    if (button.innerText == "$AVAILABLE_TEXT") {
        button.style.backgroundColor = "$AVAILABLE_COLOR";
    }
    if (button.innerText == "$SOLUTION_TEXT" ) {
        button.style.backgroundColor = "$SOLUTION_COLOR";
    }
    if (button.innerText == "$KEEPOUT_TEXT" ) {
        button.style.backgroundColor = "$KEEPOUT_COLOR";
    }
    if (button.innerText == "$PROBE_TEXT" ) {
        button.style.backgroundColor = "$PROBE_COLOR";
    }
}


const buttons = container.querySelectorAll("button");
for (let i = 0; i < buttons.length; i++) {
    const button = buttons[i];
    changeButtonStyle(button);
}

const codes = container.querySelectorAll("code");
for (let i = 0; i < codes.length; i++) {
    const code = codes[i];
    code.style.fontSize = "0.8em";
    code.style.backgroundColor = "white";
    code.style.color = "black";
    code.style.alignSelf = "center";
}


const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        const buttons = container.querySelectorAll("button");
        for (let i = 0; i < buttons.length; i++) {
            const button = buttons[i];
            changeButtonStyle(button);
        }
    });
});

observer.observe(container, { childList: true, subtree: true });
