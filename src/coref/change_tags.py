import glob, re

if __name__ == '__main__':
    for file_path in glob.glob("../../data/SentiCoref_1.0/*.tsv"):
        with open(file_path, "r") as file:
            content = file.read()

        content = content.replace("#T_CH=de.tudarmstadt.ukp.dkpro.core.api.coref.type.Coreference|referenceRelation|referenceType", "#T_CH=de.tudarmstadt.ukp.dkpro.core.api.coref.type.CoreferenceLink|referenceRelation|referenceType")
        with open(file_path, "w") as file:
            file.write(content)
