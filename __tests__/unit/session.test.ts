import TFModel from "../../src/utils/tf_model";
import * as tf from "@tensorflow/tfjs-node";
import { Sequential } from "@tensorflow/tfjs-node";

describe("Predict values", () => {
  
  it("Zero", async () => {
    const tfModel: TFModel | null = await TFModel.load("file://assets/models/trained_model/");
    if (tfModel == null) throw new Error("No model found");
    const result = tfModel.getModel().predict(tf.tensor1d([0])) as tf.Tensor;
    expect(result.dataSync()[0]).toBe(0);
  });

});