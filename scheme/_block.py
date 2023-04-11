from hashlib import blake2b
from bitstring import BitArray
import random

from utils import *
from utils import _read_data

from ._base import Scheme


class BlockScheme(Scheme):

    def __init__(self, beta, xi=1, fingerprint_bit_length=None, number_of_recipients=None):
        self.beta = beta
        self.xi = xi

        if fingerprint_bit_length is not None:
            if number_of_recipients is not None:
                super().__init__(fingerprint_bit_length, number_of_recipients)
            else:
                super().__init__(fingerprint_bit_length)
        else:
            if number_of_recipients is not None:
                super().__init__(number_of_recipients)
            else:
                super().__init__()

        self.original_relation = None
        self.binary_image = None

        self._INIT_MESSAGE = "Block scheme - initialised.\nEmbedding started...\n\tbeta: " + str(self.beta) + \
                             "\n\tfingerprint length: " + str(self.fingerprint_bit_length)

    def insertion(self, dataset_name, recipient_id, secret_key):
        print("Start Block Scheme insertion algorithm...")
        print("\tbeta: " + str(self.beta) + "\n\txi: " + str(self.xi))
        # it is assumed that the first column in the dataset is the primary key
        self.original_relation, primary_key = import_dataset_from_file(dataset_name)
        num_of_attributes = len(self.original_relation.select_dtypes(exclude='object').columns) - 1  # number of numerical attributes minus primary key

        fingerprint = super().create_fingerprint(recipient_id, secret_key)
        print("\nGenerated fingerprint for buyer " + str(recipient_id) + ": " + fingerprint.bin)
        print("Inserting the fingerprint...\n")

        fingerprinted_relation = self.original_relation.copy()
        start = time.time()

        self.binary_image = self.create_image(self.original_relation)
        shift = 10
        r0 = (secret_key << shift) + recipient_id  # threshold for pseudo random number generator
        hash_generator = blake2b()  # cryptographic hash function for generating random numbers
        j = 0  # fingerprint index

        # for each block make image in a form of list of blocks (block - list of elements)
        image_blocks = self.block_image(self.binary_image)
        print("Number of blocks: " + str(len(image_blocks)))
        print("\nInserting the fingerprint...\n")
        changes = []
        cnt = 0
        for i in range(self.get_number_of_blocks()):
            random.seed(r0)
            # r1 = random number seeded by r0 (hash value of secret key)
            r1 = random.getrandbits(32)
            # x = H(r1, recipient_id) mod beta
            hash_generator.update(((r1 << shift) + recipient_id).to_bytes(6, 'little'))
            x = int(hash_generator.hexdigest(), 32) % self.beta
            # r2 = random(r1)
            random.seed(r1)
            r2 = random.getrandbits(32)
            # y = H(r2, recipient_id) mod beta
            hash_generator.update(((r2 << shift) + recipient_id).to_bytes(6, 'little'))
            y = int(hash_generator.hexdigest(), 32) % self.beta
            # mark a bit (x,y) within the block xor fj
            block = image_blocks[i]
            mark_bit = int(block[x][y])
            # embedding fingerprint in order least significant bit -> most significant bit
            fingerprint_bit = 1 if fingerprint[j] else 0
            marked_bit = (mark_bit + fingerprint_bit) % 2
            if marked_bit != mark_bit:
                changes.append((i, x, y))
            # change the image
            image_blocks[i][x][y] = str(marked_bit)

            r0 = r2
            j += 1
            # if all fingerprint bits are used, we start again from the beginning of the fp
            if j == self.fingerprint_bit_length:
                j = 0
                cnt += 1
        print("Fingerprint bit embedded at least " + str(cnt) + " times.")

        # blocked image back to an image
        blocks_per_row = int(len(self.binary_image[0]) / self.beta)
        # iterate through rows of blocks
        h = 0
        altered_img = []
        while h < len(image_blocks) / blocks_per_row:
            # iterate through rows within the block
            for j in range(self.beta):
                row = []
                for i in range(blocks_per_row):
                    for k in range(self.beta):
                        # print("i:" + str(i + h*blocks_per_row) + " ,j:" + str(j) + " ,k:" + str(k))
                        row.append(image_blocks[i + h * blocks_per_row][j][k])  # i -> i*num of iterations
                altered_img.append(row)
            h += 1
        # add remainder of original image that was not part of any block
        # fill out missing columns
        altered_len = len(altered_img[0])
        for i, row in enumerate(altered_img):
            altered_img[i].extend(self.binary_image[i][j] for j in range(altered_len, len(self.binary_image[0])))
        # fill out missing rows
        altered_size = len(altered_img)
        altered_img.extend(self.binary_image[i] for i in range(altered_size, len(self.binary_image)))
        print("Number of bit changes: " + str(len(changes)))

        # image is ready, need to embed it back to the dataset:
        #   record the changes made to the image
        #   detect to which values from the dataset the changes apply
        #  first convert to coordinates within the big image
        changes_2d = []
        for i, x, y in changes:
            x_2d = int(i / blocks_per_row) * self.beta + x
            y_2d = int(i % blocks_per_row) * self.beta + y
            changes_2d.append((x_2d, y_2d))
        # then convert to position in the dataset
        changes_ds = [(x, int(y / self.xi), self.xi - (y % self.xi) - 1) for x, y in changes_2d]
        # change the actual values
        print("Number of changes: " + str(len(changes_ds)))
        #progress = 0
        for x, y, lsbit in changes_ds:
            # +1 for skipping the primary key
            col = self.original_relation.columns[y + 1]
            val = self.original_relation[col][x]
            val_bin = []
            val_temp = val
            if val < 0:
                val_temp = - val_temp
            while val_temp != 0:
                val_bin.append(val_temp % 2)
                val_temp = int(val_temp / 2)

            while len(val_bin) < self.xi:
                val_bin.append(0)
            # flipping the bit
            val_bin[lsbit] = (val_bin[lsbit] + 1) % 2
            # convert to int
            changed_val = 0
            for i, bit in enumerate(val_bin):
                changed_val += bit * 2 ** i
            if val < 0:
                changed_val = - changed_val
            # insert back to the dataset
            fingerprinted_relation[col][x] = changed_val
            #progress += 1

        print("Fingerprint inserted.")
        print("\tsingle fingerprint bit embedded " + str(cnt) + " times")
        # write_dataset(fingerprinted_relation, "block_scheme", dataset_name, [self.beta, self.xi], recipient_id)
        print("Time: " + str(int(time.time() - start)) + " sec.")
        return fingerprinted_relation

    def detection(self, dataset, secret_key):
        print("Start Block Scheme detection algorithm...")
        print("\tbeta: " + str(self.beta) + "\n\txi: " + str(self.xi))
        suspicious_relation = _read_data(dataset)

        if self.original_relation is None:
            self.original_relation, orig_primary_key = import_dataset(dataset)
            if self.binary_image is None:
                self.binary_image = self.create_image(self.original_relation)

        start = time.time()

        suspicious_image = self.create_image(suspicious_relation.dataframe)
        original_image = self.binary_image
        if original_image is None:
            original_image = self.create_image(self.original_relation)

        suspicious_block_img = self.block_image(suspicious_image)
        original_block_img = self.block_image(original_image)

        recipient_id = -1
        for recipient_id in range(self.number_of_recipients):
            hash_generator = blake2b()
            # for each buyer retrieve the corresponding r0
            shift = 10
            seed = (secret_key << shift) + recipient_id
            r0 = seed
            # # for each block
            f = []
            # counts for fingerprint bits
            count = [[0, 0] for x in range(self.fingerprint_bit_length)]
            for i, block in enumerate(suspicious_block_img):
                random.seed(r0)
                r1 = random.getrandbits(32)
                hash_generator.update(((r1 << shift) + recipient_id).to_bytes(6, 'little'))
                x = int(hash_generator.hexdigest(), 32) % self.beta
                # r2 = random(r1)
                random.seed(r1)
                r2 = random.getrandbits(32)
                # y = H(r2, recipient_id) mod beta
                hash_generator.update(((r2 << shift) + recipient_id).to_bytes(6, 'little'))
                y = int(hash_generator.hexdigest(), 32) % self.beta

                marking_bit = (int(block[x][y]) + int(original_block_img[i][x][y])) % 2
                f.append(marking_bit)
                count[i % self.fingerprint_bit_length][marking_bit] += 1
                r0 = r2

            # number of times the fingerprint is embedded
            omega = len(f) / self.fingerprint_bit_length
            fingerprint_template = [2] * self.fingerprint_bit_length
            # recover fingerprint
            for i in range(self.fingerprint_bit_length):
                if count[i][0] + count[i][1] == 0:
                    print("1. None suspected")
                    exit()
                # certainty of a fingerprint value
                T = 0.80
                if count[i][0] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 0
                elif count[i][1] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 1
                else:
                    print("2. None suspected")
                    exit()

            b = blake2b(key=seed.to_bytes(6, 'little'), digest_size=int(self.fingerprint_bit_length / 8))
            fingerprint = BitArray(hex=b.hexdigest())
            # from BitArray to normal array
            fingerprint_arr = [1 if bit else 0 for bit in fingerprint]
            if fingerprint_template == fingerprint_arr:
                print("Suspected buyer: " + str(recipient_id))
                break
        return recipient_id

    def create_image(self, dataset):
        """
        Creates a binary image - the collection of bits that can be modified
        :param dataset:
        :return: array of bits
        """
        print("Creating image...")
        image = []
        for r in dataset.select_dtypes(exclude='object').iterrows():
            row = 0
            for attr in r[1][1:]:
                # take xi bits and append to bit row
                if attr < 0:
                    attr = - attr
                lsbs = attr % (2 ** self.xi)  # int representation
                row = row << self.xi
                row += lsbs
            # convert to string of bits
            row = bin(row)[2:]
            while len(row) != self.xi * (len(r[1]) - 1):
                row = "0" + row
            if len(row) != self.xi * (len(r[1]) - 1):
                print("Something went wrong with creating the binary image.")
                print(len(row))
            image.append(row)
        if len(image) != len(dataset):
            print("Something went wrong with creating the binary image.")
        print("Image created!")
        return image

    def block_image(self, image):
        """
        creates a blocked representation of the image
        list of blocks (blocks - list of its elements)
        :return: list of blocks
        """
        block_img = []
        for idx in range(self.get_number_of_blocks()):
            elements = []
            # first convert index of a block to coordinates of block within the image
            block_x = int(idx / int(len(image[0]) / self.beta))
            block_y = idx % int(len(image[0]) / self.beta)
            # second, calculate elements' coordinates from block's coordinates
            for i in range(self.beta):
                row = []
                for j in range(self.beta):
                    x_el = block_x * self.beta + i
                    y_el = block_y * self.beta + j
                    row.append(image[x_el][y_el])
                elements.append(row)
            block_img.append(elements)
        return block_img

    def get_number_of_blocks(self):
        return int(len(self.binary_image[0]) / self.beta) * int(len(self.binary_image) / self.beta)