import glob, os
from pathlib import Path

def add_pos(data_type):
    train_ = f"../../data/senticoref_conll-random/{data_type}/"
    for file in os.listdir(train_):
        if file.endswith(".conll"):
            file_tsv = file.replace(".conll", ".tsv")
            mod_lines = []
            with(open(f"../../data/senticoref_pos_stanza/{file_tsv}", "r")) as pos_file:
                with(open(os.path.join(train_, file), "r")) as main_file:
                    for pos, line in zip(pos_file.readlines(), main_file.readlines()):
                        pos_parts = pos.split()
                        main_parts = line.split()
                        # print(pos_parts, main_parts)
                        if len(main_parts) > 0 and (main_parts[0] != "#begin" and main_parts[0] != "#end"):
                            main_parts[4] = pos_parts[1].replace("mte:", "").strip()
                            if main_parts[3].strip() != pos_parts[2].strip():
                                main_parts[6] = pos_parts[2].strip()
                            else:
                                main_parts[6] = "-"
                            mod_lines.append("\t".join(main_parts))
                            mod_lines.append("\n")
                        else:
                            mod_lines.append(line)
            mod_lines.append("#end document")
            Path(f"../../data/senticoref_conll-random-pos/{data_type}").mkdir(parents=True, exist_ok=True)
            with(open(f"../../data/senticoref_conll-random-pos/{data_type}/{file}", "w")) as new_file:
                new_file.writelines(mod_lines)

if __name__ == '__main__':
    add_pos("test")
    add_pos("train")
    add_pos("dev")