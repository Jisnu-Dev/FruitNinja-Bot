import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Create a BuilderConfig
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # Set workspace size to 1 MiB

        # Load the ONNX file.
        with open(onnx_file_path, 'rb') as f:
            if not parser.parse(f.read()):
                print(f"Failed to parse the ONNX file: {onnx_file_path}")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build and save the engine.
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Failed to build the TensorRT engine!")
            return None

        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"Successfully built the engine and saved to {engine_file_path}")

# Replace with your actual ONNX file path and desired engine output path
build_engine("best.onnx", "best.engine")
