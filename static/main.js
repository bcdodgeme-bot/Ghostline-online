document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const outputArea = document.querySelector("main");
    const loader = document.getElementById("loader");

    form.addEventListener("submit", function () {
        if (loader) loader.style.display = "block";
    });

    if (outputArea) {
        outputArea.scrollTop = outputArea.scrollHeight;
    }
});