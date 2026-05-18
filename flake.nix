{
  description = "Reproducible shell for dataset-artifacts cartography experiments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          commonPackages = with pkgs; [
            bashInteractive
            coreutils
            findutils
            git
            gnumake
            jq
            openssh
            pkg-config
            rsync
            uv
            python311
            stdenv.cc.cc.lib
            zlib
          ];
          nativeLibraryPath = pkgs.lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc.lib
            zlib
          ]);
          commonHook = ''
            export UV_PROJECT_ENVIRONMENT="''${UV_PROJECT_ENVIRONMENT:-.venv}"
            export UV_PYTHON="${pkgs.python311}/bin/python"
            export UV_PYTHON_DOWNLOADS="never"
            export HF_HOME="''${HF_HOME:-$PWD/.cache/huggingface}"
            export HF_DATASETS_CACHE="''${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
            export TRANSFORMERS_CACHE="''${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
            export TOKENIZERS_PARALLELISM="''${TOKENIZERS_PARALLELISM:-false}"
            export PYTHONNOUSERSITE=1
            export LD_LIBRARY_PATH="${nativeLibraryPath}:''${LD_LIBRARY_PATH:-}"
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
            fi
            if [ -d /run/opengl-driver-32/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver-32/lib:''${LD_LIBRARY_PATH:-}"
            fi
          '';
        in
        {
          default = pkgs.mkShell {
            packages = commonPackages;
            shellHook = commonHook + ''
              echo "dataset-artifacts local shell"
              echo "Run: uv sync --extra cpu --group dev"
            '';
          };

          server = pkgs.mkShell {
            packages = commonPackages;
            shellHook = commonHook + ''
              echo "dataset-artifacts GPU server shell"
              echo "Run: uv sync --extra cuda --group dev"
            '';
          };
        });
    };
}
