const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const resultCard = document.getElementById('resultCard');
const leafImage = document.getElementById('leafImage');
const label = document.getElementById('label');
const confidence = document.getElementById('confidence');
const clearBtn = document.getElementById('clearBtn');

// Click to open file picker
uploadBox.addEventListener('click', () => fileInput.click());

// Drag and drop handling
uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.style.borderColor = '#4caf50';
});
uploadBox.addEventListener('dragleave', () => {
  uploadBox.style.borderColor = '#ccc';
});
uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.style.borderColor = '#ccc';
  handleFile(e.dataTransfer.files[0]);
});

// Handle file input
fileInput.addEventListener('change', (e) => {
  handleFile(e.target.files[0]);
});

function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    // Show image
    leafImage.src = reader.result;
    uploadBox.classList.add('hidden');
    resultCard.classList.remove('hidden');

    // Simulate model prediction (replace with API call)
    setTimeout(() => {
      label.textContent = "Early Blight";
      confidence.textContent = "100.00%";
    }, 1000);
  };
  reader.readAsDataURL(file);
}

// Clear button
clearBtn.addEventListener('click', () => {
  resultCard.classList.add('hidden');
  uploadBox.classList.remove('hidden');
  fileInput.value = "";
});
