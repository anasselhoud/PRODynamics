# CentralStorage

## Main purpose
The `CentralStorage` object is defined to embody the so-called *central storage* present in **FHS** manufactoring lines to run **macro simulations**.

In the real world, the storage is **divided into two** physically different storages. Each one of them has one or many sections that can one or several sizes of products. 

For instance, the **first storage** can own two sections : the first one allowing 330 small-sized while the second one allows 25 products that can be either small or large. 


## Python structure

### Attributes
```python
self.ID = 'Central Storage'
self.env # simpy.Environment
self.manuf_line # Manufactoring line holding the central storage
self.strategy # Strategy to fill the two storages within ('stack' by default)
self.times_to_reach # Time to go to each storage (supposed different)

self.stores # Holds the items
```

The **main attribute** is `self.stores` generated using the `central_storage_config` dictionnary :

```python
{'front': [{'allowed_ref' : [],
            'capacity': int},

            {'allowed_ref' : [],
             'capacity': int},

            ...],

 'back': [{'allowed_ref' : [],
           'capacity': int},
            
            ...],
}
```

It keep tracks of the different **storages**, **blocks**, **allowed products** and **stored products**. Its final structure turns the `capacity` into a `simpy.FilterStore` to store items :

```python
{'front': [{'allowed_ref' : [],
            'store': simpy.FilterStore()},

            {'allowed_ref' : [],
            'store': simpy.FilterStore()},

            ...],

'back': [{'allowed_ref' : [],
            'store': simpy.FilterStore()},
            
            ...],
}
```


- `front` and `back` refer to the two main storages of the physical central storage. They will be called **sides** thereafter.
- Each **side** of the central storage holds a `list` of different **blocks**.
- Any block has the following characteristic :
  - A `list` of one or many **allowed products / references** : ('Ref X').
  - A `simpy.FilterStore` object that holds the items and the block capacity. [FilterStore documentation](https://simpy.readthedocs.io/en/latest/topical_guides/resources.html#stores).

In order to have **additional useful details** about the added products, items stored in `simpy.FilterStore` are dictionnaries with the following keys :

```python
{
    'name': 'Ref A', # Name of the reference
    'origin': MachineObject, # Machine from which the product comes from
    'status' : 'OK' # Not used yet.
}
```

### Methods

4 main methods have been implemented so far. They take advantage of the `simpy.FilterStore` structure in order to **check** if reference can be either put or gotten, then **put or get** one.

 ```python
def available_spot(self, ref=None) -> bool:
    """Check if there is an available spot."""

def available_ref(self, ref=None) -> bool:
    """Check if there is an available reference."""

def put(self, ref_data):
    """Try to put a reference in the storage determined by the strategy of the central storage."""

def get(self, ref=None):
    """Try to get a reference in the storage determined by the strategy of the central storage."""
 ```

 - Except from `put()` where the whole item dictionnary is required, other methods **only require the reference name** (or None when there are no restrictions).

