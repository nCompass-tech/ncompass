{
  description = "nCompass SDK";

  inputs = { nixpkgs.url = "github:nixos/nixpkgs"; };

  outputs = { self, nixpkgs } :
    let
      python_pkgs = pkgs : (with pkgs.python311Packages; [
            pip
            pudb
          ]);

      system_pkgs = pkgs : (with pkgs; [
          git-lfs
          pyright
          plantuml
        ]);

      venv_name="venv-lsp";
      venv_pip_pkgs = ''
          set -e
          
          python3 -m venv ${venv_name}
          source ${venv_name}/bin/activate
          pip install uv
          
          if [ -f "requirements.txt" ]; then
            uv pip install -r requirements.txt
          else
            echo "No requirements.txt file found"
          fi
          set +e
          '';

      free_pkgs_linux = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
        config.nvidia.acceptLicense = true;
      };
      
      free_pkgs_osx = import nixpkgs {
        system = "aarch64-darwin";
        config.allowUnfree = true;
      };
      
      linuxNixEnvPackages = pkgs : 
        pkgs.mkShell {
          buildInputs = (python_pkgs pkgs) ++ (system_pkgs pkgs);
          shellHook = 
          ''
            export PS1="$( [[ -z $IN_NIX_SHELL ]] && echo "" || echo "[$(basename $PWD)]" ) $PS1"
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ 
                                        pkgs.stdenv.cc.cc 
                                    ]}:$LD_LIBRARY_PATH
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '' + venv_pip_pkgs;
        };
      
      osxNixEnvPackages = pkgs : 
        pkgs.mkShell {
          buildInputs = (python_pkgs pkgs) ++ (system_pkgs pkgs);
          shellHook = 
          ''
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '' + venv_pip_pkgs;
        };
    
    in {
      devShell.aarch64-darwin = osxNixEnvPackages free_pkgs_osx;
      devShell.x86_64-linux = linuxNixEnvPackages free_pkgs_linux;
    };
}
