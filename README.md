# Neural Networks: Zero to Hero

Following the [tutorial series by Andrej Karpathy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

## `makemore`

The only real deviation is that I used *city names* instead of *first names* to train the model and generate from.

### Examples

Examples of generate city names, using an MLP that I implemented from scratch (including the backward pass) on ascii characters:

| Word | Generated city name          | In dataset? |
| ---- | ---------------------------- | ----------- |
| 0    | hozer                        | False       |
| 1    | walla                        | False       |
| 2    | el tonga                     | False       |
| 3    | cawatsan lo sulaw            | False       |
| 4    | antowy bayalampur            | False       |
| 5    | tocen                        | False       |
| 6    | droxde                       | False       |
| 7    | gunguo                       | False       |
| 8    | clyneffervilkoro             | False       |
| 9    | schethanicorin               | False       |
| 10   | undro woishde yet            | False       |
| 11   | pleston                      | False       |
| 12   | misika                       | False       |
| 13   | bagago                       | False       |
| 14   | toholaburito hybary          | False       |
| 15   | preihas                      | False       |
| 16   | chimas                       | False       |
| 17   | pehhanathira                 | False       |
| 18   | barmort chij                 | False       |
| 19   | baenta andsoay grrenah manga | False       |

Many of these seem nonsensical to the English eye, but the model was trained on global names, and these are actually quite representative!

E.g `Jurbarkas` and `Bhararisain` are real cities from the dataset -- but don't look familiar to me, at least.

Training on all characters of the dataset, i.e. without restricting to ascii characters, gives ~80 characters. The same model trained on this vocabularly was unable to learn anything; producing garbage output. Many of those characters are relatively rare, meaning we introduced a long-tail issue.

One direction that was interesting was also to limit the dataset to only include city names from sub-groups of countries, which share common linguistic features e.g. across Scandinavia. One challenge here, however, is preventing the model from simply memorising the dataset - generated city names mostly come out as being from dataset. Regularisation of some kind is required: smaller model, earlier stopping, random shuffling of letters, etc.


## Notes

* One limitation here is  that the dataset uses only the `ascii` alphabet, whereas the data contains many other non-English characters.
* Training on the full set of ~235 characters, ceteris paribus, the model is unable to learn anything.
  * The embedding table is 235*235 and that is larger than the embedding dimension itself
  * Bumping the embedding size and the hidden dimension also led to catastrophic performance; predicting whitespace, repeating characters, or single bigrams.

## Data sources

1. minishakespeare.csv: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
2. worldcities.csv: https://simplemaps.com/data/world-cities
