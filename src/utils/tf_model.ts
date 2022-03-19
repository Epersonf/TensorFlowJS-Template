import { Layer } from "@tensorflow/tfjs-layers/dist/engine/topology";
import * as tf from "@tensorflow/tfjs-node";
import { ModelCompileArgs, Sequential, string } from "@tensorflow/tfjs-node";

class TFModel {
  model: Sequential;
  path: string;

  static async new(path: string, createModel: (model: Sequential) => Sequential): Promise<TFModel> {
    const loadedModel = await TFModel.load(path);
    if (loadedModel != null) return loadedModel;
    return new TFModel(path, createModel(tf.sequential()));
  }

  static async load(path: string) : Promise<TFModel | null> {
    try {
      return new TFModel(path, (await tf.loadLayersModel(`${path}/model.json`)) as Sequential);
    } catch {
      return null;
    }
  }

  constructor(path: string, model: Sequential) {
    this.path = path;
    this.model = model;
  }

  getModel(): Sequential {
    return this.model;
  }

  save() {
    this.model.save(this.path);
  }
}

export default TFModel;