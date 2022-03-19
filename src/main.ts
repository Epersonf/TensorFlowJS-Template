import * as tf from "@tensorflow/tfjs-node";
import { Sequential } from "@tensorflow/tfjs-node";
import TFModel from "./utils/tf_model";

async function main() {
  const tfModel: TFModel = await TFModel.new("file://assets/models/trained_model/", (model: Sequential) => {
    model.add(tf.layers.dense({ units: 10, inputShape: [1] }));
    model.add(tf.layers.dense({ units: 1 }));
    return model;
  });
  
  tfModel.getModel().compile({
    loss: "meanSquaredError",
    optimizer: "sgd",
    metrics: ["MAE"],
  });

  const range = (start: number, stop: number): number[] => Array.from({ length: stop - start + 1 }, (_, i) => start + i);
  const vec = range(-5, 5);

  tfModel.getModel().fit(tf.tensor1d(vec), tf.tensor1d(vec.map(e => e * e)), {
    epochs: 10000
  });

  tfModel.save();
}

main();
