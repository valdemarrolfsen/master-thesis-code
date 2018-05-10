from preprocessing import label_generator

datasets = [
    '/mnt/ekstern/rubval/oslo/tiled-512x512/',
    '/mnt/ekstern/rubval/bergen/tiled-512x512/',
    '/mnt/ekstern/rubval/trondheim/tiled-512x512/',
    '/mnt/ekstern/rubval/stavanger/tiled-512x512/',
    '/mnt/ekstern/rubval/tromso/tiled-512x512/',
    '/mnt/ekstern/rubval/bodo/tiled-512x512/',
]


def run():
    for dataset in datasets:
        a = {'output': '/mnt/ekstern/rubval/vegetation',
             'input': dataset,
             'color': 1,
             'prefix': '',
             'include_empty': False,
             'binary': True,
             'res': 512,
             'class_name': 'vegetation',
             'threads': 8}
        label_generator.run(a)


if __name__ == '__main__':
    run()
