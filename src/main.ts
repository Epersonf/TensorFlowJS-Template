import * as tf from "@tensorflow/tfjs-node";
import { Sequential } from "@tensorflow/tfjs-node";
import TFModel from "./utils/tf_model";
import { plot, Plot } from "nodeplotlib";

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
  const xTrain = range(-100, 100);
  const yTrain = xTrain.map(e => e * e);

  console.log(xTrain);
  console.log(yTrain);
  
  const data: Plot[] = [
    {
      x: xTrain,
      y: yTrain,
      type: 'scatter',
    },
  ];
  
  plot(data);
  return;


  tfModel.getModel().fit(tf.tensor1d(xTrain), tf.tensor1d(yTrain), {
    epochs: 10000
  });

  tfModel.save();
}

main();
