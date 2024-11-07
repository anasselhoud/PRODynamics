# CentralStorage

## Main purpose
The `CentralStorage` object is defined to embody the so-called *central storage* present in **FHS** manufacturing lines to run **macro simulations**.

In the real world, the storage is **divided into two** physically different storages. Each one of them has one or many sections that allow one or several sizes of products. 

For instance, the **first storage** can own two sections : the first one allowing 330 small-sized while the second allows 25 products that can be either small or large. 


## Python attributes

```python
self.ID = 'Central Storage'
self.env # simpy.Environment
self.manuf_line # Manufactoring line holding the central storage
self.strategy # Strategy to fill the two storages within ('stack' by default)
self.times_to_reach # Time to go to each storage (supposed different)

self.available_stored_by_ref = # Count the number of each reference in the storage
self.available_spots_by_ref = # Count the number of available spots for each reference
self.available_routes # Holds all (origin, destination) pairs of stored items

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
- Any block has the following characteristics :
  - A `list` of one or many **allowed products / references** : `'Ref X'` for instance.
  - A `simpy.FilterStore` object that holds the items and the block capacity. See [FilterStore documentation](https://simpy.readthedocs.io/en/latest/topical_guides/resources.html#stores) for extra details.

In order to have **additional useful details** about the added products, items stored in `simpy.FilterStore` are dictionnaries with the following keys :

```python
{
    'name': 'Ref A', # Name of the reference
    'route': (OriginEntity, DestinationEntity), # Origin and destination of the item before being sent to the central storage
    'status' : 'OK' # Not used yet.
}
```

## Python methods

### Input flow
2  methods have been implemented to fill the central storage when required. 
- One takes advantage of the `simpy.FilterStore` structure in order to **check if there is an available spot** in the storage for a specified reference according to the different blocks. **Only the reference name** is required.
- The second **puts** the reference in the storage. In this case, **additional data must be provided** on top of the reference name, as detailed above.

 ```python
def available_spot(self, ref_name=None) -> bool:
    """Check if there is an available spot."""

def put(self, ref_data):
    """Try to put a reference in the storage determined by the strategy of the central storage."""
 ```

### Output flow
2  methods have been implemented to empty the central storage when required. 
- One takes advantage of the additional data stored about the items in order to **check if there is a reference with the same route** as in parameter. 
- The second **gets** the reference in the storage **based on its stored destination**. This is quite helpful to handle the transport when an item is sent back to the manufacturing line.

```python
 def available_ref_by_route(self, origin, destination) -> bool:
    """Check if there is an available reference with the same route (origin, destination)."""

def get_by_destination(self, destination):
    """Try to get a reference in the storage based on its route destination."""
```

### Unused yet
```python
def available_ref(self, ref_name=None) -> bool:
    """Check if there is an available reference."""

def get(self, ref_name=None):
    """Try to get a reference in the storage determined by the strategy of the central storage."""
```

### ⚠️ Keep in mind

This documentation is meant to explain how the object `CentralStorage` has been developped and what are its main methods.

**However**, the way the object is used is **out of the scope** of this specific documentation, **especially the conditions to decide to either send or retrieve products** from the central storage. 

This has to be defined at the level of the `ManufLine` process, not the `CentralStorage` object.
