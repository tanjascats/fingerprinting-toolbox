# Fingerprinting Relational Data
Fingerprinting toolbox is a library that allows embedding and extracting fingerprints into the relational data.

## Usage
You can use the toolbox by cloning this repository.
```
$ git clone https://github.com/tanjascats/fingerprinting-toolbox.git
```
### Fingerprint embedding (insertion)
For fingerprint insertion, we can define the scheme with the parameter gamma and bit-length of a fingerprint. The number of modified rows in the data will then be approx. #rows/gamma (TIP: use gamma to control the amount of modifications in the data). 

```
scheme = Universal(gamma=2, fingerprint_bit_length=64)
```

After the scheme is initialized, we can embedd the fingerprint using our (owner's) secret key and specifying recipient's ID: 

```
fingerprinted_data = scheme.insertion("my_data.csv", secret_key=12345678, recipient_id=0)
```

### Fingerprint extraction (detection)
For the fingerprint extraction, we provide the suspicious data and the secret key used for embedding:

```
suspect = scheme.detection("suspicious_data.csv", secret_key=12345678)
```


For more examples and detailed explanations check out the notebook [example.ipynb](https://github.com/tanjascats/fingerprinting-toolbox/blob/master/example.ipynb).
 
## Support & Contribution
The toolbox is in its early stage, but actively developing, so you can either:
- Report the issues at [Issues](https://github.com/tanjascats/fingerprinting-toolbox/issues) or
- Contact me by email: TSarcevic@sba-research.org for questions, suggestions or issues
- Pull requests are welcome
