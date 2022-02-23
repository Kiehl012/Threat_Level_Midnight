// Grab reference to user input element
const user_txt = document.getElementsByName('michael_talks');
user_txt.addEventListener("keydown", function (e) {
    if (e.code === "Enter") {
        predict();
    }
});

// Specify location of our model.json file CHANGE THIS ONCE FILES IN GITHUB ARE ORGANIZED!
const MODEL_URL = 'https://github.com/Kiehl012/Threat_Level_Midnight/blob/main/hwaijiinlee/text_gen_signature_tfjs/model.json'

// Create an asynchronous function
async function predict() {
    // Load the model
    const model = await tf.loadLayersModel(MODEL_URL);

    // Print out the architecture of the loaded model.
    // This is useful to see that it matches what we built in Python.
    console.log(model.summary());

    // Create a tensor with user input.
    const next_char = tf.constant([user_txt]);

    // Make prediction
    const states = None
    const result = [next_char]

    for (let n = 0; n < 106; n++) {
        next_char, states = model.generate_one_step(next_char, states=states)
        result.append(next_char);
    }
    
    const result = tf.strings.join(result);
    const prediction = result[0].numpy().decode('utf-8');

    return prediction;
}

// Call the prediction function
predict();