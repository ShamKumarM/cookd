import os
from typing import Optional

from supabase import create_client, Client


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set"
    )


supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)


# --------------------------------------------------
# CUSTOMER
# --------------------------------------------------

def get_customer_by_phone(phone: str):
    """
    Find a customer using their WhatsApp phone number.
    """

    response = (
        supabase
        .table("customers")
        .select("*")
        .eq("phone", phone)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]

#debugg
def debug_customers():
    response = (
        supabase
        .table("customers")
        .select("id, phone, name")
        .limit(5)
        .execute()
    )

    return response.data

#---------------------------------------------------
#Track latest order by phone
#---------------------------------------------------
def track_latest_order_by_phone(phone: str):
    """
    Find the customer by phone and return their latest order.
    """

    customer = get_customer_by_phone(phone)

    if not customer:
        return {
            "success": False,
            "error": "customer_not_found"
        }

    customer_id = customer["id"]

    order = get_latest_order(customer_id)

    if not order:
        return {
            "success": False,
            "error": "no_orders",
            "customer": customer
        }

    return {
        "success": True,
        "customer": customer,
        "order": order
    }


# --------------------------------------------------
# ORDERS
# --------------------------------------------------

def get_customer_orders(customer_id: str):
    """
    Get all orders belonging to a customer.
    """

    response = (
        supabase
        .table("orders")
        .select("*")
        .eq("customer_id", customer_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def get_latest_order(customer_id: str):
    """
    Get the customer's most recent order.
    """

    response = (
        supabase
        .table("orders")
        .select("*")
        .eq("customer_id", customer_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]


def get_order_status(
    customer_id: str,
    order_id: Optional[str] = None,
):
    """
    Get order status.

    If order_id is supplied, retrieve that order.
    Otherwise retrieve the latest order.
    """

    query = (
        supabase
        .table("orders")
        .select("*")
        .eq("customer_id", customer_id)
    )

    if order_id:
        query = query.eq("id", order_id)
    else:
        query = query.order("created_at", desc=True).limit(1)

    response = query.execute()

    if not response.data:
        return None

    return response.data[0]


# --------------------------------------------------
# PRODUCTS
# --------------------------------------------------

def get_product_stock(product_id: str):
    """
    Get current product inventory.
    """

    response = (
        supabase
        .table("products")
        .select(
            "id, sku, name, price_inr, stock_qty, active"
        )
        .eq("id", product_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]


def search_products(query: str):
    """
    Search products by name.

    This is a simple database lookup.
    Semantic product search remains the job of ChromaDB.
    """

    response = (
        supabase
        .table("products")
        .select(
            "id, sku, name, category, pack_size, "
            "price_inr, spice_level, stock_qty, active"
        )
        .ilike("name", f"%{query}%")
        .eq("active", True)
        .execute()
    )

    return response.data

