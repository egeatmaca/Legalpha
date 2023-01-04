const tf = require('@tensorflow/tfjs-node');
const PorterStemmer = require('porter-stemmer').PorterStemmer;

const xs_train = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys_train = tf.tensor2d([1, 1, 0, 0], [4, 1]);

const xs_test = tf.tensor2d([1, 3, 5], [3, 1]);
const ys_test = tf.tensor2d([1, 0, 0], [3, 1]);

function tokenize(text) {
    return text.split(" ");
}

function stem(word) {
  const porterStemmer = new PorterStemmer();
  return porterStemmer.stem(text);
}

async function train() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({
      loss: "binaryCrossentropy",
      optimizer: "adam",
      metrics: ["accuracy"],
    });

    await model.fit(xs_train, ys_train, { epochs: 10 });

    const results = model.evaluate(xs_test, ys_test);
    for (let result of results) {
        result.print();
    }

    model.save("file://app/model");
}

train();