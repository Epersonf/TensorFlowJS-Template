import TFModel from "../../src/utils/tf_model";
import * as tf from "@tensorflow/tfjs-node";
import { Sequential } from "@tensorflow/tfjs-node";

describe("Predict values", () => {
  
  it("Zero", async () => {
    const tfModel: TFModel | null = await TFModel.load("file://assets/models/trained_model/");
    if (tfModel == null) throw new Error("No model found");
    expect(tfModel.predict(0)).toBe(0);
  });

});